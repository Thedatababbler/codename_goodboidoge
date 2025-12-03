"""
offline_treatment_adaptation.py

Offline Adaptation for Treatment/Medication Tasks based on ACE Framework.

适用场景：
- 治疗方案制定 (Treatment Planning)
- 药物选择与给药 (Medication Selection & Dosing)
- 治疗流程/步骤规划 (Treatment Workflow)

与诊断任务的区别：
1. 已知疾病诊断，直接使用诊断名召回对应的治疗guideline
2. 不需要更新 Retrieval Signatures（检索已经较容易）
3. 关注治疗方案的正确性、安全性、指南依从性
4. Delta Update 主要针对治疗建议、药物信息、注意事项等

流程：
Diagnosis -> Retriever -> Generator(Treatment) -> Reflector -> Curator -> Delta Update

TODO: 
- [ ] 准备治疗类任务的数据集
- [ ] 实现具体的评估逻辑 (TreatmentEvaluator)
- [ ] 添加药物数据库集成
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict

# ACE 基础组件
from ace import (
    OpenAIClient, DummyLLMClient, Generator, Retriever,
    GuidelinePlaybook, DeltaBatch, DeltaOperation, GeneratorOutput,
    LLMClient,
)
from ace.playbook import GuidelineBullet, GuidelineSection

# 治疗专用 prompts
from ace.treatment_prompts import (
    TreatmentTaskType,
    TREATMENT_GENERATOR_PROMPT,
    TREATMENT_REFLECTOR_PROMPT,
    TREATMENT_CURATOR_PROMPT,
    MEDICATION_GENERATOR_PROMPT,
    MEDICATION_REFLECTOR_PROMPT,
    get_prompts_for_task,
)

# 复用医疗 Reflector/Curator 的核心逻辑
from ace.medical_roles import (
    BulletTag,
    ProposedOperation,
    MedicalReflectorOutput,
    MedicalCuratorOutput,
    apply_bullet_tags,
    _safe_json_loads,
)


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class TreatmentCase:
    """治疗任务的病例数据结构"""
    case_id: str
    diagnosis: str                          # 已确诊的疾病
    patient_info: Dict[str, Any]            # 患者信息
    question: str                           # 治疗问题/任务
    ground_truth: Optional[str] = None      # 标准治疗方案（如果有）
    task_type: str = TreatmentTaskType.TREATMENT_PLAN
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreatmentCase":
        return cls(
            case_id=str(data.get("case_id", "")),
            diagnosis=str(data.get("diagnosis", "")),
            patient_info=data.get("patient_info", {}),
            question=str(data.get("question", "")),
            ground_truth=data.get("ground_truth"),
            task_type=data.get("task_type", TreatmentTaskType.TREATMENT_PLAN),
            metadata=data.get("metadata", {}),
        )


@dataclass 
class TreatmentGeneratorOutput:
    """治疗任务 Generator 的输出"""
    reasoning: str
    treatment_plan: str                     # 或 medication 等
    key_considerations: str
    bullet_ids: List[str]
    raw: Dict[str, Any]
    
    # 药物任务特有字段
    medication: Optional[Dict[str, str]] = None
    alternatives: Optional[List[str]] = None
    monitoring: Optional[List[str]] = None


@dataclass
class TreatmentReflectorOutput:
    """治疗任务 Reflector 的输出"""
    reasoning: str
    error_identification: str
    safety_concerns: str
    guideline_adherence: str
    key_insight: str
    bullet_tags: List[BulletTag]
    proposed_operations: List[ProposedOperation]
    raw: Dict[str, Any]
    
    def to_delta_batch(self) -> DeltaBatch:
        ops = [op.to_delta_operation() for op in self.proposed_operations]
        return DeltaBatch(operations=ops, created_by="treatment-reflector")


# ============================================================================
# Treatment Generator
# ============================================================================

class TreatmentGenerator:
    """
    治疗任务的 Generator。
    支持多种任务类型：治疗方案、药物选择、治疗流程。
    """
    
    def __init__(
        self,
        llm: LLMClient,
        task_type: str = TreatmentTaskType.TREATMENT_PLAN,
        prompt_template: Optional[str] = None,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.task_type = task_type
        self.prompt_template = prompt_template or get_prompts_for_task(task_type)[0]
        self.max_retries = max_retries
    
    def generate(
        self,
        *,
        case: TreatmentCase,
        playbook: GuidelinePlaybook,
        **kwargs: Any,
    ) -> TreatmentGeneratorOutput:
        """
        生成治疗方案。
        
        Parameters
        ----------
        case : TreatmentCase
            治疗任务病例
        playbook : GuidelinePlaybook
            治疗指南 playbook
        
        Returns
        -------
        TreatmentGeneratorOutput
        """
        # 构建 context
        context = self._build_context(case)
        
        # 构建 prompt
        playbook_json = playbook.to_hierarchical_json(include_timestamps=False)
        prompt = self.prompt_template.format(
            playbook=playbook_json,
            context=context,
            question=case.question,
        )
        
        # 调用 LLM
        last_error = None
        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                return self._parse_output(data)
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                # 重试
                prompt += "\n\nPlease return valid JSON only."
        
        raise RuntimeError("TreatmentGenerator failed") from last_error
    
    def _build_context(self, case: TreatmentCase) -> str:
        """构建患者上下文"""
        lines = [
            f"Diagnosis: {case.diagnosis}",
        ]
        
        # 添加患者信息
        for key, value in case.patient_info.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def _parse_output(self, data: Dict[str, Any]) -> TreatmentGeneratorOutput:
        """解析 LLM 输出"""
        bullet_ids = [
            str(item) for item in data.get("bullet_ids", [])
            if isinstance(item, (str, int))
        ]
        
        return TreatmentGeneratorOutput(
            reasoning=str(data.get("reasoning", "")),
            treatment_plan=str(data.get("treatment_plan", "")),
            key_considerations=str(data.get("key_considerations", "")),
            bullet_ids=bullet_ids,
            raw=data,
            medication=data.get("medication"),
            alternatives=data.get("alternatives"),
            monitoring=data.get("monitoring"),
        )


# ============================================================================
# Treatment Reflector
# ============================================================================

class TreatmentReflector:
    """
    治疗任务的 Reflector。
    分析治疗方案的正确性、安全性、指南依从性。
    
    与诊断任务的区别：
    - 不生成 retrieval_patterns（已知诊断，检索容易）
    - 关注安全性、药物相互作用、禁忌症等
    """
    
    def __init__(
        self,
        llm: LLMClient,
        task_type: str = TreatmentTaskType.TREATMENT_PLAN,
        prompt_template: Optional[str] = None,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.task_type = task_type
        self.prompt_template = prompt_template or get_prompts_for_task(task_type)[1]
        self.max_retries = max_retries
    
    def reflect(
        self,
        *,
        case: TreatmentCase,
        generator_output: TreatmentGeneratorOutput,
        playbook: GuidelinePlaybook,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> TreatmentReflectorOutput:
        """
        反思治疗方案。
        """
        playbook_excerpt = self._make_playbook_excerpt(playbook, generator_output.bullet_ids)
        playbook_structure = self._make_playbook_structure(playbook)
        
        # 构建 prompt - 根据任务类型填充不同字段
        format_kwargs = {
            "question": case.question,
            "diagnosis": case.diagnosis,
            "reasoning": generator_output.reasoning,
            "ground_truth": case.ground_truth or "(not available)",
            "playbook_title": playbook.title,
            "playbook_id": playbook.guideline_id,
            "playbook_excerpt": playbook_excerpt,
            "playbook_structure": playbook_structure,
            "feedback": feedback or "(none)",
        }
        
        # 根据任务类型添加特定字段
        if self.task_type == TreatmentTaskType.TREATMENT_PLAN:
            format_kwargs["treatment_plan"] = generator_output.treatment_plan
            format_kwargs["key_considerations"] = generator_output.key_considerations
        elif self.task_type == TreatmentTaskType.MEDICATION:
            format_kwargs["medication"] = json.dumps(generator_output.medication or {})
            format_kwargs["alternatives"] = json.dumps(generator_output.alternatives or [])
            format_kwargs["contraindications_checked"] = generator_output.key_considerations
            format_kwargs["monitoring"] = json.dumps(generator_output.monitoring or [])
        
        prompt = self.prompt_template.format(**format_kwargs)
        
        # 调用 LLM
        last_error = None
        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                return self._parse_output(data)
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                prompt += "\n\nPlease return valid JSON only."
        
        raise RuntimeError("TreatmentReflector failed") from last_error
    
    def _parse_output(self, data: Dict[str, Any]) -> TreatmentReflectorOutput:
        """解析 Reflector 输出"""
        # 解析 bullet_tags
        bullet_tags = []
        for item in data.get("bullet_tags", []):
            if isinstance(item, dict) and "id" in item and "tag" in item:
                bullet_tags.append(BulletTag(id=str(item["id"]), tag=str(item["tag"]).lower()))
        
        # 解析 proposed_operations
        proposed_ops = []
        for item in data.get("proposed_operations", []):
            if isinstance(item, dict) and "type" in item:
                proposed_ops.append(ProposedOperation.from_dict(item))
        
        return TreatmentReflectorOutput(
            reasoning=str(data.get("reasoning", "")),
            error_identification=str(data.get("error_identification", "")),
            safety_concerns=str(data.get("safety_concerns", "")),
            guideline_adherence=str(data.get("guideline_adherence", "")),
            key_insight=str(data.get("key_insight", "")),
            bullet_tags=bullet_tags,
            proposed_operations=proposed_ops,
            raw=data,
        )
    
    def _make_playbook_excerpt(self, playbook: GuidelinePlaybook, bullet_ids: List[str]) -> str:
        lines = []
        for bid in bullet_ids:
            bullet = playbook.get_bullet(bid)
            if bullet:
                lines.append(f"[{bullet.id}] {bullet.content[:150]}...")
        return "\n".join(lines) if lines else "(no bullets referenced)"
    
    def _make_playbook_structure(self, playbook: GuidelinePlaybook) -> str:
        lines = []
        for section in playbook.sections():
            lines.append(f"Section [{section.id}]: {section.title}")
            for bid in section.bullet_ids[:3]:
                bullet = playbook.get_bullet(bid)
                if bullet:
                    lines.append(f"  - [{bullet.id}] {bullet.content[:80]}...")
        return "\n".join(lines) if lines else "(empty)"


# ============================================================================
# Treatment Curator
# ============================================================================

class TreatmentCurator:
    """
    治疗任务的 Curator。
    执行 Delta Update，不更新 Retrieval Signatures。
    """
    
    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = TREATMENT_CURATOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries
    
    def curate_direct(
        self,
        reflection: TreatmentReflectorOutput,
        playbook: GuidelinePlaybook,
        *,
        auto_apply: bool = False,
    ) -> MedicalCuratorOutput:
        """
        直接应用 Reflector 的建议（不经过 LLM 二次验证）。
        
        不更新 Retrieval Signatures（与诊断任务的区别）。
        """
        all_ops = []
        skipped = []
        
        # 添加 Reflector 提出的操作
        for op in reflection.proposed_operations:
            all_ops.append(op.to_delta_operation())
        
        # 将 bullet_tags 转换为 TAG_BULLET 操作
        for tag in reflection.bullet_tags:
            all_ops.append(DeltaOperation(
                type="TAG_BULLET",
                bullet_id=tag.id,
                metadata={tag.tag: 1},
            ))
        
        delta = DeltaBatch(operations=all_ops, created_by="treatment-curator")
        
        applied = []
        if auto_apply:
            result = playbook.apply_delta(delta)
            applied = result.get("applied", [])
            skipped = result.get("skipped", [])
        
        return MedicalCuratorOutput(
            reasoning="Direct application of treatment reflector proposals",
            delta=delta,
            raw={"proposals_count": len(reflection.proposed_operations)},
            applied_ops=applied,
            skipped_ops=skipped,
        )


# ============================================================================
# 评估器接口（待实现）
# ============================================================================

class TreatmentEvaluator(ABC):
    """治疗方案评估器的抽象基类"""
    
    @abstractmethod
    def evaluate(
        self,
        case: TreatmentCase,
        generator_output: TreatmentGeneratorOutput,
    ) -> Dict[str, Any]:
        """
        评估生成的治疗方案。
        
        Returns
        -------
        Dict with:
            - is_correct: bool
            - feedback: str
            - metrics: Dict[str, float]
        """
        pass


class SimpleTreatmentEvaluator(TreatmentEvaluator):
    """
    简单的治疗方案评估器（基于字符串匹配）。
    
    TODO: 实现更复杂的评估逻辑：
    - 基于关键药物/治疗步骤的匹配
    - 基于 LLM 的语义评估
    - 基于医学知识库的验证
    """
    
    def evaluate(
        self,
        case: TreatmentCase,
        generator_output: TreatmentGeneratorOutput,
    ) -> Dict[str, Any]:
        gt = case.ground_truth or ""
        pred = generator_output.treatment_plan
        
        if not gt:
            return {
                "is_correct": None,  # 无法评估
                "feedback": "No ground truth available",
                "metrics": {},
            }
        
        # 简单匹配（需要改进）
        gt_lower = gt.lower()
        pred_lower = pred.lower()
        
        # 提取关键词进行匹配
        is_correct = any(word in pred_lower for word in gt_lower.split() if len(word) > 3)
        
        feedback = "Correct" if is_correct else f"Expected: {gt}"
        
        return {
            "is_correct": is_correct,
            "feedback": feedback,
            "metrics": {"simple_match": 1.0 if is_correct else 0.0},
        }


# ============================================================================
# 主要的 Offline Adaptation Pipeline
# ============================================================================

class TreatmentOfflineAdapter:
    """
    治疗任务的离线适配器。
    
    特点：
    1. 使用诊断名直接召回治疗指南（已知诊断）
    2. 不更新 Retrieval Signatures
    3. Delta Update 专注于治疗建议、药物信息等
    4. 支持多种任务类型
    """
    
    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        task_type: str = TreatmentTaskType.TREATMENT_PLAN,
        *,
        evaluator: Optional[TreatmentEvaluator] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.task_type = task_type
        
        # 获取对应的 prompts
        gen_prompt, ref_prompt, cur_prompt = get_prompts_for_task(task_type)
        
        # 初始化 Agents
        self.generator = TreatmentGenerator(llm, task_type, gen_prompt)
        self.reflector = TreatmentReflector(llm, task_type, ref_prompt)
        self.curator = TreatmentCurator(llm, cur_prompt)
        self.evaluator = evaluator or SimpleTreatmentEvaluator()
        
        # Playbook 缓存
        self._playbook_cache: Dict[str, GuidelinePlaybook] = {}
        
        # 统计
        self._stats = {
            "total_cases": 0,
            "evaluated_cases": 0,
            "correct_cases": 0,
            "delta_operations": 0,
        }
    
    def get_or_create_playbook(self, diagnosis: str) -> Optional[GuidelinePlaybook]:
        """
        根据诊断名获取治疗指南 playbook。
        
        使用诊断名直接召回（已知诊断，检索容易）。
        """
        if diagnosis in self._playbook_cache:
            return self._playbook_cache[diagnosis]
        
        result = self.retriever.retrieve_exact(diagnosis)
        if not result.entries:
            # TODO: 尝试模糊匹配或使用通用治疗指南
            return None
        
        entry = result.entries[0]
        playbook = GuidelinePlaybook.from_markdown(
            guideline_id=f"treatment_{diagnosis.replace(' ', '_')}",
            title=f"Treatment Guidelines: {entry.title}",
            text=entry.text,
        )
        
        self._playbook_cache[diagnosis] = playbook
        return playbook
    
    def process_case(
        self,
        case: TreatmentCase,
        *,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        处理单个治疗任务病例。
        
        流程: Generator -> Evaluator -> Reflector -> Curator -> Delta Update
        """
        # 获取 playbook
        playbook = self.get_or_create_playbook(case.diagnosis)
        if playbook is None:
            return {"error": f"No guideline for diagnosis: {case.diagnosis}"}
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"[CASE] {case.case_id}")
            print(f"[DIAGNOSIS] {case.diagnosis}")
            print(f"[TASK] {case.task_type}")
            print(f"{'='*50}")
        
        # Step 1: Generate
        generator_output = self.generator.generate(
            case=case,
            playbook=playbook,
        )
        
        if verbose:
            print(f"\n[GENERATOR]")
            print(f"  Treatment: {generator_output.treatment_plan[:100]}...")
        
        # Step 2: Evaluate
        eval_result = self.evaluator.evaluate(case, generator_output)
        is_correct = eval_result.get("is_correct")
        feedback = eval_result.get("feedback", "")
        
        if verbose:
            status = "✅" if is_correct else ("❓" if is_correct is None else "❌")
            print(f"\n[EVAL] {status} {feedback[:50]}...")
        
        # Step 3: Reflect
        reflector_output = self.reflector.reflect(
            case=case,
            generator_output=generator_output,
            playbook=playbook,
            feedback=feedback,
        )
        
        if verbose:
            print(f"\n[REFLECTOR]")
            print(f"  Insight: {reflector_output.key_insight[:80]}...")
            print(f"  Ops: {len(reflector_output.proposed_operations)}")
        
        # Step 4: Curate & Apply Delta
        curator_output = self.curator.curate_direct(
            reflection=reflector_output,
            playbook=playbook,
            auto_apply=True,
        )
        
        if verbose:
            print(f"\n[CURATOR]")
            print(f"  Applied: {len(curator_output.applied_ops)}")
        
        # 更新统计
        self._stats["total_cases"] += 1
        if is_correct is not None:
            self._stats["evaluated_cases"] += 1
            if is_correct:
                self._stats["correct_cases"] += 1
        self._stats["delta_operations"] += len(curator_output.applied_ops)
        
        return {
            "case_id": case.case_id,
            "diagnosis": case.diagnosis,
            "generator_output": generator_output,
            "eval_result": eval_result,
            "reflector_output": reflector_output,
            "curator_output": curator_output,
        }
    
    def run_offline_adaptation(
        self,
        cases: List[TreatmentCase],
        *,
        epochs: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        运行离线适配。
        
        Parameters
        ----------
        cases : List[TreatmentCase]
            治疗任务病例列表
        epochs : int
            训练轮数
        verbose : bool
            是否打印详细信息
        """
        all_results = []
        
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"\n{'#'*50}")
                print(f"# EPOCH {epoch}/{epochs}")
                print(f"{'#'*50}")
            
            for i, case in enumerate(cases, 1):
                if verbose:
                    print(f"\n--- Case {i}/{len(cases)} ---")
                
                result = self.process_case(case, verbose=verbose)
                all_results.append(result)
        
        # 计算准确率
        accuracy = (
            self._stats["correct_cases"] / self._stats["evaluated_cases"]
            if self._stats["evaluated_cases"] > 0 else 0
        )
        
        return {
            "results": all_results,
            "stats": self._stats.copy(),
            "accuracy": accuracy,
            "playbooks": {
                k: v.to_hierarchical_dict()
                for k, v in self._playbook_cache.items()
            },
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()


# ============================================================================
# 示例用法（框架，待数据准备后完善）
# ============================================================================

def main():
    """
    示例主函数（框架）。
    
    TODO:
    - [ ] 准备治疗类任务的数据集
    - [ ] 实现更完善的评估逻辑
    - [ ] 添加药物数据库集成
    """
    
    # 示例病例（待替换为真实数据）
    example_cases = [
        TreatmentCase(
            case_id="TX001",
            diagnosis="Influenza",
            patient_info={
                "age": 35,
                "gender": "female",
                "allergies": ["penicillin"],
                "current_medications": [],
            },
            question="What is the recommended treatment for this patient with confirmed influenza?",
            ground_truth="Oseltamivir 75mg twice daily for 5 days",
            task_type=TreatmentTaskType.TREATMENT_PLAN,
        ),
        TreatmentCase(
            case_id="MED001",
            diagnosis="Hypertension",
            patient_info={
                "age": 55,
                "gender": "male",
                "comorbidities": ["type 2 diabetes"],
                "current_medications": ["metformin"],
            },
            question="Select appropriate first-line antihypertensive medication.",
            ground_truth="ACE inhibitor (e.g., lisinopril) or ARB",
            task_type=TreatmentTaskType.MEDICATION,
        ),
    ]
    
    print("="*60)
    print("TREATMENT OFFLINE ADAPTATION - FRAMEWORK")
    print("="*60)
    print("\nThis is a framework script. To run:")
    print("1. Prepare your treatment task dataset")
    print("2. Initialize LLM client (OpenAI or local)")
    print("3. Initialize Retriever with treatment guidelines")
    print("4. Run adapter.run_offline_adaptation(cases)")
    print("\nExample case structure:")
    print(json.dumps({
        "case_id": "TX001",
        "diagnosis": "Disease name",
        "patient_info": {"age": 35, "allergies": []},
        "question": "Treatment question",
        "ground_truth": "Expected treatment",
        "task_type": "treatment_plan | medication | workflow",
    }, indent=2))
    
    # 框架代码（取消注释以运行）
    # llm = OpenAIClient(model="gpt-4")
    # retriever = Retriever(json_path="data/guideline_dict.json", source="wikidoc")
    # adapter = TreatmentOfflineAdapter(llm, retriever, task_type=TreatmentTaskType.TREATMENT_PLAN)
    # results = adapter.run_offline_adaptation(example_cases, epochs=1)


if __name__ == "__main__":
    main()

