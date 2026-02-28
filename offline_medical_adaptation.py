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
import pickle
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

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
# 检查点数据结构（用于断点恢复）
# ---------------------------------------------------------------------

@dataclass
class Checkpoint:
    """检查点数据，用于保存和恢复训练状态"""
    current_epoch: int = 1
    current_disease_idx: int = 0
    current_case_idx: int = 0
    diseases_order: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    accuracy_per_epoch: List[float] = field(default_factory=list)
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    ops_log_lines: List[str] = field(default_factory=list)
    epoch_correct: int = 0
    epoch_total: int = 0
    completed: bool = False
    # Playbook 缓存数据（序列化后的 dict 格式）
    playbook_cache_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def save(self, filepath: str):
        """保存检查点到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[CHECKPOINT] Saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'Checkpoint':
        """从文件加载检查点"""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"[CHECKPOINT] Loaded from {filepath}")
        print(f"  - Epoch: {checkpoint.current_epoch}")
        print(f"  - Disease: {checkpoint.current_disease_idx}/{len(checkpoint.diseases_order)}")
        print(f"  - Case: {checkpoint.current_case_idx}")
        return checkpoint


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
        max_playbook_tokens: Optional[int] = 6000,
    ):
        self.llm = llm
        self.retriever = retriever
        self.max_playbook_tokens = max_playbook_tokens

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

    def serialize_playbook_cache(self) -> Dict[str, Dict[str, Any]]:
        """序列化 playbook 缓存为可保存的 dict 格式（使用 to_dict 与 from_dict 匹配）"""
        return {
            disease: pb.to_dict()
            for disease, pb in self._playbook_cache.items()
        }

    def restore_playbook_cache(self, cache_data: Dict[str, Dict[str, Any]]) -> None:
        """从序列化数据恢复 playbook 缓存"""
        for disease, pb_data in cache_data.items():
            self._playbook_cache[disease] = GuidelinePlaybook.from_dict(pb_data)
    
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
        
        # Step 3: Generator推理（带截断以避免 token 超限）
        playbook_json = playbook.to_hierarchical_json(
            include_timestamps=False,
            max_tokens=self.max_playbook_tokens,
        )
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
        save_ops_log: bool = True,
        checkpoint_path: Optional[str] = None,
        resume_from: Optional[str] = None,
        save_checkpoint_every: int = 10,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> Dict[str, Any]:
        """
        运行离线适配循环。

        支持：
        - 多个不同疾病的病例
        - 多轮训练（epochs）
        - 自动维护每个疾病的playbook
        - 断点恢复（checkpoint）
        - API 调用重试

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
        save_ops_log : bool
            是否保存 bullet 操作日志到 txt 文件
        checkpoint_path : Optional[str]
            检查点文件保存路径（默认为 output_dir/checkpoint.pkl）
        resume_from : Optional[str]
            从指定的检查点文件恢复
        save_checkpoint_every : int
            每处理多少个 case 保存一次检查点
        max_retries : int
            API 调用失败时的最大重试次数
        retry_delay : float
            重试之间的等待时间（秒）

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

        # 确保疾病顺序一致（用于断点恢复）
        diseases_order = sorted(cases_by_disease.keys())

        # 设置检查点路径
        if checkpoint_path is None:
            checkpoint_path = os.path.join(output_dir, "checkpoint.pkl")

        # 初始化或恢复检查点
        if resume_from and os.path.exists(resume_from):
            checkpoint = Checkpoint.load(resume_from)
            # 恢复统计数据
            self._stats = checkpoint.stats.copy()
            all_results = checkpoint.all_results
            accuracy_per_epoch = checkpoint.accuracy_per_epoch
            ops_log_lines = checkpoint.ops_log_lines
            start_epoch = checkpoint.current_epoch
            start_disease_idx = checkpoint.current_disease_idx
            start_case_idx = checkpoint.current_case_idx
            epoch_correct = checkpoint.epoch_correct
            epoch_total = checkpoint.epoch_total
            # 恢复 playbook 缓存
            if hasattr(checkpoint, 'playbook_cache_data') and checkpoint.playbook_cache_data:
                self.restore_playbook_cache(checkpoint.playbook_cache_data)
                print(f"[RESUME] Restored {len(checkpoint.playbook_cache_data)} playbooks from cache")
            print(f"\n[RESUME] Continuing from Epoch {start_epoch}, Disease {start_disease_idx}, Case {start_case_idx}")
        else:
            checkpoint = Checkpoint(diseases_order=diseases_order)
            all_results = []
            accuracy_per_epoch = []
            ops_log_lines = []
            start_epoch = 1
            start_disease_idx = 0
            start_case_idx = 0
            epoch_correct = 0
            epoch_total = 0

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# OFFLINE ADAPTATION")
            print(f"# Total cases: {len(cases)}")
            print(f"# Unique diseases: {len(cases_by_disease)}")
            print(f"# Epochs: {epochs}")
            if resume_from:
                print(f"# Resuming from checkpoint")
            print(f"{'#'*60}")

        total_processed = 0

        for epoch in range(start_epoch, epochs + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch}/{epochs}")
                print(f"{'='*60}")

            # 如果是恢复的 epoch，使用恢复的计数，否则重置
            if epoch > start_epoch:
                epoch_correct = 0
                epoch_total = 0
                start_disease_idx = 0
                start_case_idx = 0

            # 按疾病处理（确保同一疾病的病例使用同一playbook - Delta Update）
            for disease_idx, disease in enumerate(diseases_order):
                # 跳过已处理的疾病
                if epoch == start_epoch and disease_idx < start_disease_idx:
                    continue

                disease_cases = cases_by_disease[disease]

                if verbose:
                    print(f"\n--- Disease: {disease} ({len(disease_cases)} cases) ---")

                for case_idx, case in enumerate(disease_cases):
                    # 跳过已处理的 case
                    if epoch == start_epoch and disease_idx == start_disease_idx and case_idx < start_case_idx:
                        continue

                    if verbose:
                        print(f"\n  Case {case_idx + 1}/{len(disease_cases)}")

                    # 带重试的 process_case
                    result = None
                    for retry in range(max_retries):
                        try:
                            result = self.process_case(case, verbose=verbose)
                            break  # 成功，跳出重试循环
                        except Exception as e:
                            if retry < max_retries - 1:
                                print(f"\n[ERROR] Failed to process case: {e}")
                                print(f"[RETRY] Waiting {retry_delay}s before retry {retry + 2}/{max_retries}...")
                                time.sleep(retry_delay)
                            else:
                                print(f"\n[ERROR] Max retries exceeded. Saving checkpoint and exiting.")
                                # 保存检查点
                                checkpoint.current_epoch = epoch
                                checkpoint.current_disease_idx = disease_idx
                                checkpoint.current_case_idx = case_idx
                                checkpoint.stats = self._stats.copy()
                                checkpoint.all_results = all_results
                                checkpoint.accuracy_per_epoch = accuracy_per_epoch
                                checkpoint.ops_log_lines = ops_log_lines
                                checkpoint.epoch_correct = epoch_correct
                                checkpoint.epoch_total = epoch_total
                                checkpoint.playbook_cache_data = self.serialize_playbook_cache()
                                checkpoint.save(checkpoint_path)
                                raise RuntimeError(f"Failed after {max_retries} retries: {e}")

                    if result is None:
                        continue

                    all_results.append(result)

                    # 记录 bullet 操作日志
                    if save_ops_log and "curator_output" in result:
                        curator_output = result["curator_output"]
                        ops_log_lines.append(f"\n{'='*60}")
                        ops_log_lines.append(f"Disease: {disease}")
                        ops_log_lines.append(f"Case {case_idx + 1}/{len(disease_cases)} | Epoch {epoch}")
                        ops_log_lines.append(f"Prediction: {result.get('prediction', 'N/A')}")
                        ops_log_lines.append(f"Ground Truth: {result.get('ground_truth', 'N/A')}")
                        ops_log_lines.append(f"Correct: {'✅' if result.get('is_correct') else '❌'}")
                        ops_log_lines.append(f"{'='*60}")
                        ops_log_lines.append(f"Applied Operations ({len(curator_output.applied_ops)}):")
                        for op in curator_output.applied_ops:
                            ops_log_lines.append(f"  ✓ {op}")
                        if curator_output.skipped_ops:
                            ops_log_lines.append(f"Skipped Operations ({len(curator_output.skipped_ops)}):")
                            for op in curator_output.skipped_ops:
                                ops_log_lines.append(f"  ✗ {op}")

                    epoch_total += 1
                    if result.get("is_correct"):
                        epoch_correct += 1

                    total_processed += 1

                    # 定期保存检查点
                    if total_processed % save_checkpoint_every == 0:
                        checkpoint.current_epoch = epoch
                        checkpoint.current_disease_idx = disease_idx
                        checkpoint.current_case_idx = case_idx + 1  # 下一个 case
                        checkpoint.stats = self._stats.copy()
                        checkpoint.all_results = all_results
                        checkpoint.accuracy_per_epoch = accuracy_per_epoch
                        checkpoint.ops_log_lines = ops_log_lines
                        checkpoint.epoch_correct = epoch_correct
                        checkpoint.epoch_total = epoch_total
                        checkpoint.playbook_cache_data = self.serialize_playbook_cache()
                        checkpoint.save(checkpoint_path)

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

        # 保存 bullet 操作日志
        if save_ops_log and ops_log_lines:
            os.makedirs(output_dir, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filepath = os.path.join(output_dir, f"bullet_ops_log_{timestamp}.txt")
            with open(log_filepath, "w", encoding="utf-8") as f:
                f.write(f"Bullet Operations Log\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Total Cases: {len(cases)}\n")
                f.write(f"Epochs: {epochs}\n")
                f.write(f"Total Delta Operations: {self._stats['delta_operations']}\n")
                f.write("\n".join(ops_log_lines))
            if verbose:
                print(f"[SAVED] {log_filepath}")

        # 标记完成并删除检查点
        checkpoint.completed = True
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            if verbose:
                print(f"[CHECKPOINT] Removed completed checkpoint: {checkpoint_path}")

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
        """
        诊断匹配检查 - 使用灵活的匹配策略

        支持:
        1. 精确匹配
        2. 包含匹配
        3. 归一化匹配（忽略标点符号差异如 - vs /）
        4. 缩写匹配（如 ADHD = Attention-deficit hyperactivity disorder）
        """
        if not ground_truth:
            return False

        pred = prediction.strip().lower()
        gt = ground_truth.strip().lower()

        # 1. 精确匹配或包含匹配
        if (pred == gt) or (gt in pred) or (pred in gt):
            return True

        # 2. 归一化匹配：统一标点符号后比较
        def normalize(s: str) -> str:
            """归一化字符串：移除/替换标点，统一空格"""
            import re
            # 将常见分隔符统一为空格
            s = re.sub(r'[-/\\()，,;:：；]', ' ', s)
            # 移除多余空格
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        pred_norm = normalize(pred)
        gt_norm = normalize(gt)

        if (pred_norm == gt_norm) or (gt_norm in pred_norm) or (pred_norm in gt_norm):
            return True

        # 3. 缩写匹配：常见医学缩写映射
        abbreviations = {
            'adhd': ['attention deficit hyperactivity disorder', 'attention-deficit hyperactivity disorder', 'attention-deficit/hyperactivity disorder'],
            'ptsd': ['post traumatic stress disorder', 'post-traumatic stress disorder'],
            'copd': ['chronic obstructive pulmonary disease'],
            'gerd': ['gastroesophageal reflux disease'],
            'ckd': ['chronic kidney disease'],
            'hiv': ['human immunodeficiency virus'],
            'aids': ['acquired immunodeficiency syndrome', 'acquired immune deficiency syndrome'],
            'mi': ['myocardial infarction'],
            'dvt': ['deep vein thrombosis', 'deep venous thrombosis'],
            'pe': ['pulmonary embolism'],
            'cva': ['cerebrovascular accident', 'stroke'],
            'tia': ['transient ischemic attack'],
            'sle': ['systemic lupus erythematosus'],
            'ra': ['rheumatoid arthritis'],
            'ms': ['multiple sclerosis'],
            'als': ['amyotrophic lateral sclerosis'],
            'gbs': ['guillain barre syndrome', 'guillain-barre syndrome', 'guillain-barré syndrome'],
            'ibs': ['irritable bowel syndrome'],
            'ibd': ['inflammatory bowel disease'],
            'uti': ['urinary tract infection'],
            'cad': ['coronary artery disease'],
            'chf': ['congestive heart failure'],
            'afib': ['atrial fibrillation'],
            'dm': ['diabetes mellitus'],
            't1dm': ['type 1 diabetes mellitus', 'type 1 diabetes'],
            't2dm': ['type 2 diabetes mellitus', 'type 2 diabetes'],
            'htn': ['hypertension'],
            'bph': ['benign prostatic hyperplasia'],
            'osa': ['obstructive sleep apnea'],
            'ocd': ['obsessive compulsive disorder', 'obsessive-compulsive disorder'],
            'gad': ['generalized anxiety disorder'],
            'mdd': ['major depressive disorder'],
            'bpd': ['borderline personality disorder'],
        }

        # 检查预测或真值是否为缩写
        for abbr, full_forms in abbreviations.items():
            # 如果 pred 是缩写，gt 是全称
            if pred_norm == abbr or abbr in pred_norm.split():
                for full in full_forms:
                    if normalize(full) == gt_norm or normalize(full) in gt_norm or gt_norm in normalize(full):
                        return True
            # 如果 gt 是缩写，pred 是全称
            if gt_norm == abbr or abbr in gt_norm.split():
                for full in full_forms:
                    if normalize(full) == pred_norm or normalize(full) in pred_norm or pred_norm in normalize(full):
                        return True

        # 4. 音调符号处理：移除重音符号后比较（如 é -> e）
        def remove_accents(s: str) -> str:
            """移除重音符号"""
            import unicodedata
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

        pred_ascii = remove_accents(pred_norm)
        gt_ascii = remove_accents(gt_norm)

        if (pred_ascii == gt_ascii) or (gt_ascii in pred_ascii) or (pred_ascii in gt_ascii):
            return True

        # 5. 模糊匹配：去除常见后缀词后比较
        def remove_common_suffixes(s: str) -> str:
            """移除常见后缀词"""
            suffixes = ['disorder', 'disease', 'syndrome', 'condition', 'infection', 'type']
            words = s.split()
            filtered = [w for w in words if w not in suffixes]
            return ' '.join(filtered)

        pred_core = remove_common_suffixes(pred_ascii)
        gt_core = remove_common_suffixes(gt_ascii)

        if pred_core and gt_core:
            if (pred_core == gt_core) or (gt_core in pred_core) or (pred_core in gt_core):
                return True

        return False
    
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
# 数据加载工具
# ---------------------------------------------------------------------

def load_cases(source) -> List[Dict[str, Any]]:
    """
    加载病例数据，自动适配两种格式：
    1. 直接的 case list (如 test_cases)
    2. jsonl 文件路径 (如 medqa_train2k.jsonl)

    Parameters
    ----------
    source : list or str
        病例列表或 jsonl 文件路径

    Returns
    -------
    List[Dict[str, Any]]
        统一格式的病例列表
    """
    if isinstance(source, list):
        # 直接传入的 list (如 test_cases)
        return source
    elif isinstance(source, str):
        # jsonl 文件路径
        cases = []
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # 如果是嵌套结构，提取 OSCE_Examination
                if "OSCE_Examination" in data:
                    cases.append(data["OSCE_Examination"])
                else:
                    cases.append(data)
        return cases
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")


# ---------------------------------------------------------------------
# 示例用法
# ---------------------------------------------------------------------

def main():
    """主函数示例"""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()  # 加载 .env 文件

    parser = argparse.ArgumentParser(description="Medical Offline Adaptation")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to jsonl file (e.g., medqa_train2k.jsonl). If not provided, uses built-in test_cases.")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Maximum number of cases to process (for testing)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--output-dir", type=str, default="output/playbooks",
                        help="Directory to save playbooks")
    parser.add_argument("--max-playbook-tokens", type=int, default=6000,
                        help="Maximum tokens for playbook to avoid context overflow (default: 6000)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint file (e.g., output/playbooks/run_xxx/checkpoint.pkl)")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N cases (default: 10)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries for API calls (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=5.0,
                        help="Delay between retries in seconds (default: 5.0)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for LLM (gpt-5-nano only supports 1.0)")
    args = parser.parse_args()

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
    model_name = os.getenv("OPENAI_MODEL")
    if not model_name:
        raise ValueError("OPENAI_MODEL environment variable is required. Please set it in .env file.")
    print(f"[INFO] Using model: {model_name}")

    try:
        # 确定 temperature（gpt-5-nano 只支持 1.0）
        temperature = args.temperature if args.temperature is not None else 0.0
        llm = OpenAIClient(
            model=model_name,
            temperature=temperature,
            max_output_tokens=4096,  # Reflector需要足够的token来输出完整的JSON
        )
        print(f"[INFO] Temperature: {temperature}")
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
    adapter = MedicalOfflineAdapter(
        llm,
        retriever,
        max_playbook_tokens=args.max_playbook_tokens,
    )
    print(f"[INFO] Max playbook tokens: {args.max_playbook_tokens}")

    # 加载病例数据
    if args.data:
        print(f"\n[INFO] Loading cases from: {args.data}")
        cases = load_cases(args.data)
    else:
        print("\n[INFO] Using built-in test_cases")
        cases = test_cases

    # 限制病例数量（用于测试）
    if args.max_cases and args.max_cases < len(cases):
        print(f"[INFO] Limiting to first {args.max_cases} cases")
        cases = cases[:args.max_cases]

    print(f"[INFO] Total cases to process: {len(cases)}")

    # 创建带时间戳的输出目录（或使用恢复路径）
    from datetime import datetime
    if args.resume:
        # 从 checkpoint 路径提取输出目录
        output_dir_with_timestamp = os.path.dirname(args.resume)
        print(f"[INFO] Resuming to output directory: {output_dir_with_timestamp}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_with_timestamp = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(output_dir_with_timestamp, exist_ok=True)
        print(f"[INFO] Output directory: {output_dir_with_timestamp}")

    # 运行离线适配
    print("\n" + "="*60)
    print("OFFLINE ADAPTATION")
    print("="*60)

    results = adapter.run_offline_adaptation(
        cases,
        epochs=args.epochs,
        verbose=True,
        save_playbooks=True,
        output_dir=output_dir_with_timestamp,
        resume_from=args.resume,
        save_checkpoint_every=args.checkpoint_every,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    
    # 打印最终统计
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Final Accuracy: {results['final_accuracy']:.2%}")
    print(f"Total Delta Operations: {results['stats']['delta_operations']}")
    print(f"Retrieval Patterns Added: {results['stats']['retrieval_patterns_added']}")

    # 打印每个 epoch 的准确率变化（进化报告）
    if len(results['accuracy_per_epoch']) > 1:
        print(f"\n{'='*60}")
        print("EVOLUTION REPORT (Accuracy per Epoch)")
        print(f"{'='*60}")
        for i, acc in enumerate(results['accuracy_per_epoch'], 1):
            bar = '█' * int(acc * 30) + '░' * (30 - int(acc * 30))
            print(f"  Epoch {i}: {acc:.2%} |{bar}|")

        # 计算提升
        first_acc = results['accuracy_per_epoch'][0]
        last_acc = results['accuracy_per_epoch'][-1]
        improvement = last_acc - first_acc
        if improvement > 0:
            print(f"\n  📈 Improvement: +{improvement:.2%} (from {first_acc:.2%} to {last_acc:.2%})")
        elif improvement < 0:
            print(f"\n  📉 Decline: {improvement:.2%} (from {first_acc:.2%} to {last_acc:.2%})")
        else:
            print(f"\n  ➡️  No change: {first_acc:.2%}")

    # 保存进化报告到文件
    evolution_log = os.path.join(output_dir_with_timestamp, "evolution_report.txt")
    with open(evolution_log, "w", encoding="utf-8") as f:
        f.write("Evolution Report\n")
        f.write(f"{'='*40}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Cases: {len(cases)}\n")
        f.write(f"Epochs: {args.epochs}\n\n")
        f.write("Accuracy per Epoch:\n")
        for i, acc in enumerate(results['accuracy_per_epoch'], 1):
            f.write(f"  Epoch {i}: {acc:.2%}\n")
        if len(results['accuracy_per_epoch']) > 1:
            improvement = results['accuracy_per_epoch'][-1] - results['accuracy_per_epoch'][0]
            f.write(f"\nTotal Improvement: {improvement:+.2%}\n")
    print(f"[SAVED] {evolution_log}")
    
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
