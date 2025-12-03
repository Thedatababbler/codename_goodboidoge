"""
offline_medical_adaptation.py

Offline Adaptation for Medical Guideline Playbooks based on ACE Framework.

核心设计思想：
1. **GT召回策略**：在offline adaptation阶段，使用病例的ground truth诊断直接召回对应的guideline playbook
   - 避免用症状RAG精度不高的问题
   - 将offline阶段视为"训练"阶段

2. **Retrieval Signatures Section**：每个playbook新增一个专门的section
   - 用于存储从病例中提取的检索特征（症状组合、风险因素等）
   - 使得后续能从症状信息更精准地召回guideline

3. **Delta Update**：所有更新都是增量的
   - 一个疾病的guideline对应多个病例
   - 每次处理只添加/更新/标记特定的bullets
   - 不会全量替换playbook

流程：
Retriever(GT) -> Generator -> MedicalReflector -> MedicalCurator -> Delta Update
"""

import json
import os
from typing import Dict, Any, List, Optional
from collections import defaultdict

from ace import (
    OpenAIClient, Playbook, DummyLLMClient, Generator, 
    OfflineAdapter, Sample, TaskEnvironment, EnvironmentResult, Retriever,
    GuidelinePlaybook, DeltaBatch, DeltaOperation, GeneratorOutput,
)
from ace.medical_roles import (
    MedicalReflector, MedicalCurator, MedicalReflectorOutput, MedicalCuratorOutput,
    apply_bullet_tags, create_medical_agents,
    MEDICAL_REFLECTOR_PROMPT, MEDICAL_CURATOR_PROMPT,
    # Retrieval Section 相关
    RETRIEVAL_SECTION_ID, RETRIEVAL_SECTION_TITLE,
    ensure_retrieval_section, get_retrieval_signatures, initialize_playbook_for_medical,
)
from utils.utils import build_diagnosis_question


# ---------------------------------------------------------------------
# Generator Prompt 模板
# ---------------------------------------------------------------------

GENERATOR_PROMPT_MEDICAL = (
    "You are a clinical reasoning agent working on diagnosis-only tasks. "
    "Use the provided playbook as domain knowledge. "
    "From the task context and the query, output ONE plausible single best diagnosis.\n"
    "\n"
    "================ PLAYBOOK (domain knowledge) ================\n"
    "{playbook}\n"
    "=============================================================\n"
    "\n"
    "================ TASK CONTEXT (structured vignette) =========\n"
    "{context}\n"
    "=============================================================\n"
    "\n"
    "================ QUERY (what to produce) ====================\n"
    "{question}\n"
    "=============================================================\n"
    "\n"
    "REQUIREMENTS:\n"
    "1) Return EXACTLY ONE valid JSON object and NOTHING ELSE.\n"
    "2) JSON keys (all required):\n"
    "   - \"reasoning\": a SHORT justification (1–3 sentences) citing key evidence.\n"
    "   - \"final_answer\": the single most likely diagnosis as a short phrase.\n"
    "   - \"bullet_ids\": an array of playbook bullet IDs you referenced; [] if none.\n"
    "3) Use DOUBLE quotes for all JSON keys/strings.\n"
    "\n"
    "OUTPUT JSON:"
)


# ---------------------------------------------------------------------
# 医疗任务环境
# ---------------------------------------------------------------------

class MedicalDiagnosisEnv(TaskEnvironment):
    """医疗诊断任务环境"""
    
    def evaluate(self, sample: Sample, generator_output: GeneratorOutput) -> EnvironmentResult:
        gt = sample.ground_truth or ""
        pred = generator_output.final_answer.strip().lower()
        gt_lower = gt.lower().strip()
        
        correct = (pred == gt_lower) or (gt_lower in pred) or (pred in gt_lower)
        
        if correct:
            feedback = f"✅ Correct. Predicted '{generator_output.final_answer}' matches '{gt}'."
            metrics = {"accuracy": 1.0}
        else:
            feedback = f"❌ Incorrect. Expected: '{gt}', got: '{generator_output.final_answer}'."
            metrics = {"accuracy": 0.0}
        
        return EnvironmentResult(feedback=feedback, ground_truth=gt, metrics=metrics)


# ---------------------------------------------------------------------
# 主要的 Offline Adaptation Pipeline
# ---------------------------------------------------------------------

class MedicalOfflineAdapter:
    """
    医疗Guideline离线适配器。
    
    核心特性：
    1. GT召回：使用ground truth诊断名直接召回playbook（避免RAG精度问题）
    2. Retrieval Signatures：自动维护检索特征section
    3. Delta Update：所有更新都是增量的，支持多病例训练同一playbook
    4. Playbook缓存：同一疾病的多个病例共享同一个playbook实例
    """
    
    def __init__(
        self,
        llm: Any,
        retriever: Retriever,
        *,
        generator_prompt: str = GENERATOR_PROMPT_MEDICAL,
        reflector_prompt: str = MEDICAL_REFLECTOR_PROMPT,
        curator_prompt: str = MEDICAL_CURATOR_PROMPT,
    ):
        self.llm = llm
        self.retriever = retriever
        
        # 初始化各Agent
        self.generator = Generator(llm, prompt_template=generator_prompt)
        self.reflector = MedicalReflector(llm, prompt_template=reflector_prompt)
        self.curator = MedicalCurator(llm, prompt_template=curator_prompt)
        
        # Playbook缓存：disease_name -> GuidelinePlaybook
        # 确保同一疾病的多个病例共享同一个playbook实例（Delta Update）
        self._playbook_cache: Dict[str, GuidelinePlaybook] = {}
        
        # 训练统计
        self._stats = {
            "total_cases": 0,
            "correct_cases": 0,
            "delta_operations": 0,
            "retrieval_patterns_added": 0,
        }
    
    # ----------------------------------------------------------------
    # Playbook 管理（支持GT召回和Delta Update）
    # ----------------------------------------------------------------
    
    def get_or_create_playbook(
        self, 
        disease_name: str,
        *,
        force_refresh: bool = False,
    ) -> Optional[GuidelinePlaybook]:
        """
        获取或创建疾病的playbook。
        
        使用GT（疾病名）直接召回，避免RAG精度问题。
        同一疾病的多个病例共享同一个playbook实例。
        
        Parameters
        ----------
        disease_name : str
            Ground Truth 诊断名称
        force_refresh : bool
            是否强制重新加载（忽略缓存）
        
        Returns
        -------
        GuidelinePlaybook or None
        """
        # 检查缓存
        if not force_refresh and disease_name in self._playbook_cache:
            return self._playbook_cache[disease_name]
        
        # 使用GT（疾病名）直接召回guideline
        result = self.retriever.retrieve_exact(disease_name)
        if not result.entries:
            print(f"[WARNING] No guideline found for GT: '{disease_name}'")
            return None
        
        entry = result.entries[0]
        playbook = GuidelinePlaybook.from_markdown(
            guideline_id=f"{entry.source}_{disease_name.replace(' ', '_')}",
            title=entry.title,
            text=entry.text,
        )
        
        # 初始化Retrieval Signatures section
        initialize_playbook_for_medical(playbook, disease_name)
        
        # 缓存
        self._playbook_cache[disease_name] = playbook
        return playbook
    
    def get_cached_playbook(self, disease_name: str) -> Optional[GuidelinePlaybook]:
        """获取缓存的playbook（不触发检索）"""
        return self._playbook_cache.get(disease_name)
    
    # ----------------------------------------------------------------
    # 单病例处理
    # ----------------------------------------------------------------
    
    def process_case(
        self,
        case: Dict[str, Any],
        *,
        use_gt_retrieval: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        处理单个病例，执行完整的 Generate -> Reflect -> Curate -> Delta Update 流程。
        
        Parameters
        ----------
        case : Dict[str, Any]
            病例数据，必须包含 'Correct_Diagnosis' 或 'CorrectDiagnosis'
        use_gt_retrieval : bool
            是否使用GT召回（推荐在offline阶段使用）
        verbose : bool
            是否打印详细信息
        
        Returns
        -------
        Dict with processing results
        """
        # 提取GT
        ground_truth = case.get("Correct_Diagnosis") or case.get("CorrectDiagnosis") or ""
        if not ground_truth:
            return {"error": "Case missing ground truth diagnosis", "case": case}
        
        # Step 1: 使用GT召回playbook（Delta Update - 复用已有playbook）
        if use_gt_retrieval:
            playbook = self.get_or_create_playbook(ground_truth)
        else:
            # 可以扩展为使用症状RAG
            playbook = self.get_or_create_playbook(ground_truth)
        
        if playbook is None:
            return {"error": f"No guideline found for '{ground_truth}'", "case": case}
        
        # Step 2: 构建问题
        question = build_diagnosis_question(case)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[CASE] GT: {ground_truth}")
            print(f"[PLAYBOOK] {playbook.title} (ID: {playbook.guideline_id})")
            print(f"[RETRIEVAL SIGNATURES] {len(get_retrieval_signatures(playbook))} patterns")
            print(f"{'='*60}")
        
        # Step 3: Generator推理
        playbook_json = playbook.to_hierarchical_json(include_timestamps=False)
        generator_output = self.generator.generate(
            question=question,
            context="",
            playbook=playbook_json,  # type: ignore[arg-type]
        )
        
        if verbose:
            print(f"\n[GENERATOR]")
            print(f"  Prediction: {generator_output.final_answer}")
            print(f"  Reasoning: {generator_output.reasoning[:150]}...")
        
        # Step 4: 评估
        is_correct = self._check_diagnosis(generator_output.final_answer, ground_truth)
        feedback = "Correct diagnosis." if is_correct else f"Incorrect. Expected: {ground_truth}"
        
        if verbose:
            print(f"\n[EVAL] {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        
        # Step 5: Reflector分析（生成Delta Update建议）
        reflector_output = self.reflector.reflect(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
        )
        
        if verbose:
            print(f"\n[REFLECTOR]")
            print(f"  Key Insight: {reflector_output.key_insight[:100]}..." if reflector_output.key_insight else "  No new insights")
            print(f"  Retrieval Patterns: {len(reflector_output.retrieval_patterns)}")
            for p in reflector_output.retrieval_patterns[:3]:
                print(f"    - {p[:60]}...")
            print(f"  Proposed Ops: {len(reflector_output.proposed_operations)}")
        
        # Step 6: Curator执行Delta Update
        curator_output = self.curator.curate_direct(
            reflection=reflector_output,
            playbook=playbook,
            auto_apply=True,  # 自动应用delta
            deduplicate_retrieval=True,  # 对retrieval patterns去重
        )
        
        if verbose:
            print(f"\n[CURATOR - DELTA UPDATE]")
            print(f"  Applied: {len(curator_output.applied_ops)} ops")
            for op in curator_output.applied_ops[:5]:
                print(f"    ✓ {op}")
            if curator_output.skipped_ops:
                print(f"  Skipped: {len(curator_output.skipped_ops)} ops (duplicates/invalid)")
        
        # 更新统计
        self._stats["total_cases"] += 1
        if is_correct:
            self._stats["correct_cases"] += 1
        self._stats["delta_operations"] += len(curator_output.applied_ops)
        self._stats["retrieval_patterns_added"] += len([
            op for op in curator_output.applied_ops 
            if RETRIEVAL_SECTION_ID in op
        ])
        
        return {
            "ground_truth": ground_truth,
            "prediction": generator_output.final_answer,
            "is_correct": is_correct,
            "generator_output": generator_output,
            "reflector_output": reflector_output,
            "curator_output": curator_output,
            "playbook_stats": {
                "sections": len(playbook.sections()),
                "bullets": len(playbook.bullets()),
                "retrieval_signatures": len(get_retrieval_signatures(playbook)),
            },
        }
    
    # ----------------------------------------------------------------
    # 批量处理（多病例多轮）
    # ----------------------------------------------------------------
    
    def run_offline_adaptation(
        self,
        cases: List[Dict[str, Any]],
        *,
        epochs: int = 1,
        verbose: bool = True,
        save_playbooks: bool = True,
        output_dir: str = "output/playbooks",
    ) -> Dict[str, Any]:
        """
        运行离线适配循环。
        
        支持：
        - 多个不同疾病的病例
        - 多轮训练（epochs）
        - 自动维护每个疾病的playbook
        
        Parameters
        ----------
        cases : List[Dict[str, Any]]
            病例列表，每个病例必须包含ground truth
        epochs : int
            训练轮数
        verbose : bool
            是否打印详细信息
        save_playbooks : bool
            是否保存最终的playbooks
        output_dir : str
            playbook保存目录
        
        Returns
        -------
        Dict with training statistics and final playbooks
        """
        # 按疾病分组病例
        cases_by_disease: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for case in cases:
            gt = case.get("Correct_Diagnosis") or case.get("CorrectDiagnosis") or ""
            if gt:
                cases_by_disease[gt].append(case)
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# OFFLINE ADAPTATION")
            print(f"# Total cases: {len(cases)}")
            print(f"# Unique diseases: {len(cases_by_disease)}")
            print(f"# Epochs: {epochs}")
            print(f"{'#'*60}")
        
        all_results = []
        accuracy_per_epoch = []
        
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch}/{epochs}")
                print(f"{'='*60}")
            
            epoch_correct = 0
            epoch_total = 0
            
            # 按疾病处理（确保同一疾病的病例使用同一playbook - Delta Update）
            for disease, disease_cases in cases_by_disease.items():
                if verbose:
                    print(f"\n--- Disease: {disease} ({len(disease_cases)} cases) ---")
                
                for i, case in enumerate(disease_cases, 1):
                    if verbose:
                        print(f"\n  Case {i}/{len(disease_cases)}")
                    
                    result = self.process_case(case, verbose=verbose)
                    all_results.append(result)
                    
                    epoch_total += 1
                    if result.get("is_correct"):
                        epoch_correct += 1
            
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            accuracy_per_epoch.append(epoch_accuracy)
            
            if verbose:
                print(f"\n[EPOCH {epoch} SUMMARY]")
                print(f"  Accuracy: {epoch_accuracy:.2%} ({epoch_correct}/{epoch_total})")
                print(f"  Total Delta Ops: {self._stats['delta_operations']}")
                print(f"  Retrieval Patterns Added: {self._stats['retrieval_patterns_added']}")
        
        # 保存playbooks
        if save_playbooks:
            os.makedirs(output_dir, exist_ok=True)
            for disease, playbook in self._playbook_cache.items():
                safe_name = disease.replace(" ", "_").replace("/", "_")
                filepath = os.path.join(output_dir, f"{safe_name}_playbook.json")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(playbook.to_hierarchical_json())
                if verbose:
                    print(f"[SAVED] {filepath}")
        
        return {
            "results": all_results,
            "accuracy_per_epoch": accuracy_per_epoch,
            "final_accuracy": accuracy_per_epoch[-1] if accuracy_per_epoch else 0,
            "stats": self._stats.copy(),
            "playbooks": {
                disease: pb.to_hierarchical_dict()
                for disease, pb in self._playbook_cache.items()
            },
        }
    
    # ----------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------
    
    def _check_diagnosis(self, prediction: str, ground_truth: str) -> bool:
        """诊断匹配检查"""
        if not ground_truth:
            return False
        pred = prediction.strip().lower()
        gt = ground_truth.strip().lower()
        return (pred == gt) or (gt in pred) or (pred in gt)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        return self._stats.copy()
    
    def get_all_playbooks(self) -> Dict[str, GuidelinePlaybook]:
        """获取所有缓存的playbooks"""
        return self._playbook_cache.copy()
    
    def export_retrieval_signatures(self) -> Dict[str, List[str]]:
        """导出所有疾病的retrieval signatures（用于后续检索）"""
        return {
            disease: get_retrieval_signatures(pb)
            for disease, pb in self._playbook_cache.items()
        }


# ---------------------------------------------------------------------
# 示例用法
# ---------------------------------------------------------------------

def main():
    """主函数示例"""
    
    # 测试病例
    test_cases = [
        {
            "Objective_for_Doctor": "Evaluate and diagnose the patient presenting with a chronic lesion on the lower lip.",
            "Patient_Actor": {
                "Demographics": "58-year-old white male",
                "History": "3-month history of painless lesion on lower lip. 20-year smoking history, outdoor worker.",
                "Symptoms": {"Primary_Symptom": "Painless lesion on the lower lip", "Secondary_Symptoms": []},
                "Past_Medical_History": "Hypertension, type 2 diabetes.",
                "Social_History": "Smokes 1 pack/day, works outdoors.",
            },
            "Physical_Examination_Findings": {
                "Oral_Examination": {"Oral_Cavity": "Single ulcer near vermillion border, hard base, non-tender."}
            },
            "Test_Results": {"Biopsy": {"Findings": "Squamous cell carcinoma confirmed."}},
            "Correct_Diagnosis": "Squamous cell carcinoma"
        },
        {
            "Objective_for_Doctor": "Evaluate patient with respiratory symptoms.",
            "Patient_Actor": {
                "Demographics": "35-year-old female",
                "History": "3-day history of fever, cough, and body aches.",
                "Symptoms": {"Primary_Symptom": "High fever and dry cough", "Secondary_Symptoms": ["myalgia", "fatigue"]},
            },
            "Physical_Examination_Findings": {
                "Vital_Signs": {"Temperature": "39.2°C", "Heart_Rate": "102 bpm"}
            },
            "Test_Results": {"Rapid_Flu_Test": {"Findings": "Positive for Influenza A"}},
            "Correct_Diagnosis": "Influenza"
        }
    ]
    
    # 初始化LLM
    try:
        llm = OpenAIClient(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=0.0,
            max_output_tokens=1024,
        )
    except Exception as e:
        print(f"[WARNING] OpenAI init failed: {e}")
        print("[INFO] Using DummyLLMClient...")
        llm = create_dummy_llm_for_demo()
    
    # 初始化Retriever
    retriever = Retriever(
        json_path="data/guideline_dict.json",
        source="wikidoc",
        match_fields="title",
    )
    
    # 创建适配器
    adapter = MedicalOfflineAdapter(llm, retriever)
    
    # 运行离线适配
    print("\n" + "="*60)
    print("OFFLINE ADAPTATION DEMO")
    print("="*60)
    
    results = adapter.run_offline_adaptation(
        test_cases,
        epochs=1,
        verbose=True,
        save_playbooks=True,
        output_dir="output/playbooks",
    )
    
    # 打印最终统计
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Final Accuracy: {results['final_accuracy']:.2%}")
    print(f"Total Delta Operations: {results['stats']['delta_operations']}")
    print(f"Retrieval Patterns Added: {results['stats']['retrieval_patterns_added']}")
    
    # 导出retrieval signatures（供后续检索使用）
    retrieval_sigs = adapter.export_retrieval_signatures()
    print(f"\nRetrieval Signatures by Disease:")
    for disease, sigs in retrieval_sigs.items():
        print(f"  {disease}: {len(sigs)} patterns")
        for sig in sigs[:3]:
            print(f"    - {sig[:60]}...")


def create_dummy_llm_for_demo():
    """创建演示用的DummyLLMClient"""
    client = DummyLLMClient()
    
    # Generator响应
    client.queue(json.dumps({
        "reasoning": "Classic presentation with risk factors and biopsy confirmation.",
        "bullet_ids": [],
        "final_answer": "squamous cell carcinoma"
    }))
    
    # Reflector响应
    client.queue(json.dumps({
        "reasoning": "Correct diagnosis based on classic presentation.",
        "error_identification": "None",
        "root_cause_analysis": "N/A",
        "correct_approach": "Risk factor assessment + biopsy confirmation.",
        "key_insight": "Chronic non-healing lip lesions + smoking + sun exposure = consider SCC.",
        "retrieval_patterns": [
            "Painless lip ulcer + smoking history → consider squamous cell carcinoma",
            "Chronic sun exposure + lip lesion + hard base → SCC workup needed"
        ],
        "bullet_tags": [],
        "proposed_operations": []
    }))
    
    # 第二个case的响应
    client.queue(json.dumps({
        "reasoning": "Fever, cough, myalgia with positive rapid test confirms influenza.",
        "bullet_ids": [],
        "final_answer": "influenza"
    }))
    
    client.queue(json.dumps({
        "reasoning": "Classic flu presentation with confirmatory test.",
        "error_identification": "None",
        "root_cause_analysis": "N/A",
        "correct_approach": "Clinical presentation + rapid testing.",
        "key_insight": "Sudden onset fever + myalgia + respiratory symptoms during flu season.",
        "retrieval_patterns": [
            "High fever + dry cough + myalgia + fatigue → consider influenza",
            "Sudden onset respiratory illness + body aches → influenza workup"
        ],
        "bullet_tags": [],
        "proposed_operations": []
    }))
    
    return client


if __name__ == "__main__":
    main()
