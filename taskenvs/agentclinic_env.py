# examples/med_env.py
# -*- coding: utf-8 -*-
"""
MedicalDiagnosisEnv: 一个适配 ACE-open 的医疗诊断环境，用于诊断型任务。
- evaluate(): 基于 "Correct_Diagnosis" 进行判分并给出结构化反馈，供 Reflector 使用
- make_samples_from_osce(): 将你给定的 OSCE JSON（同一格式的多病例也可）转换为 Sample 列表
"""

from __future__ import annotations
import json, re
from typing import Dict, List, Tuple, Any, Iterable, Optional
from ace import TaskEnvironment, EnvironmentResult, Sample

# --------- 文本规范化与同义词处理 ---------

_DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    # 你可以持续扩充：左边是规范写法，右边是等价/常见写法
    "acute appendicitis": [
        "appendicitis", "acute-appendicitis", "acute_appencitis", "acute appy", "appy"
    ],
    "community-acquired pneumonia": [
        "pneumonia", "cap", "community acquired pneumonia"
    ],
    "acute pharyngitis": [
        "pharyngitis", "sore throat", "acute sore throat"
    ],
    "acute bronchitis": [
        "bronchitis", "acute-bronchitis"
    ],
}

def _normalize(s: str) -> str:
    s = s.strip().lower()
    # 去标点 & 多空格
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _canonicalize(ans: str, synonyms: Dict[str, List[str]]) -> Tuple[str, str]:
    """
    返回 (canonical, raw_norm)
    - raw_norm: 规范化后原字符串
    - canonical: 若命中同义词映射则返回规范写法，否则返回 raw_norm
    """
    raw_norm = _normalize(ans)
    # 先 exact，再包含式（避免长短写差异）
    for canon, vars_ in synonyms.items():
        if raw_norm == _normalize(canon):
            return (_normalize(canon), raw_norm)
        for v in vars_:
            if raw_norm == _normalize(v):
                return (_normalize(canon), raw_norm)
    # 轻度包含式（谨慎）：
    for canon, vars_ in synonyms.items():
        names = [canon] + vars_
        if any(_normalize(n) in raw_norm or raw_norm in _normalize(n) for n in names):
            return (_normalize(canon), raw_norm)
    return (raw_norm, raw_norm)

def _is_correct(pred: str, gold: str, synonyms: Dict[str, List[str]]) -> Tuple[bool, str, str]:
    c_pred, n_pred = _canonicalize(pred, synonyms)
    c_gold, n_gold = _canonicalize(gold, synonyms)
    return (c_pred == c_gold, c_pred, c_gold)

# --------- 环境本体 ---------

class MedicalDiagnosisEnv(TaskEnvironment):
    """
    医疗诊断离线评测环境（AppWorld 风格的最小可用版本）：
    - 输入：Sample(question=临床小结, ground_truth=诊断标签, context=原始结构化字段)
    - 评测：取 Generator 的 final_answer，与 gold 做等价匹配；返回带“证据提示”的反馈字符串。
    - 设计点：反馈里包含『应考虑的关键证据点』，便于 Reflector 生成 violation/adherence 条目。
    """

    def __init__(
        self,
        *,
        synonyms: Optional[Dict[str, List[str]]] = None,
        strict: bool = False,
    ) -> None:
        """
        synonyms: 自定义/扩展的诊断同义词映射；默认含常见条目
        strict:   严格模式（只 exact 同义词映射），否则启用轻度包含式匹配
        """
        self.synonyms = synonyms or _DEFAULT_SYNONYMS
        self.strict = strict

    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        """
        与 ACE 的 Adapter 对齐：
          - ground_truth: 从 sample.ground_truth 读取（创建 Sample 时已经放入）
          - generator_output.final_answer: 模型诊断输出
        反馈策略：
          - 正确：返回 "correct" 带关键证据提示（帮助 Reflector 强化遵循）
          - 错误：返回 "expected <gold> but got <pred>" 以及需考虑的关键证据点（帮助反思器定位违反之处）
        """
        gold = (sample.ground_truth or "").strip()
        pred = (getattr(generator_output, "final_answer", "") or "").strip()

        ok, c_pred, c_gold = _is_correct(pred, gold, self.synonyms)

        # 从 context 中抽取“证据点”，帮助 Reflector 产出结构化反思（遵循/违反）
        hints = self._collect_key_evidence(sample.context)

        if ok:
            feedback = (
                f"correct: diagnosis={gold}. "
                f"adherence: considered key evidence -> {hints or 'N/A'}"
            )
        else:
            feedback = (
                f"expected {gold} but got {pred}. "
                f"violation: key evidence to consider -> {hints or 'N/A'}"
            )

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=gold,
        )

    # ---------- 证据提示（从结构化字段生成，供 Reflector 使用） ----------

    def _collect_key_evidence(self, ctx: Any) -> str:
        """
        从 Sample.context（保存 OSCE 结构体）中提炼若干“应考虑的证据线索”
        这些线索将写入 feedback，指导 Reflector 做条款/要点对齐（等价于 guideline anchors 的雏形）
        """
        if not ctx or not isinstance(ctx, dict):
            return ""
        p = ctx.get("Patient_Actor", {})
        hx = p.get("History", "")
        symptoms = p.get("Symptoms", {})
        pe = ctx.get("Physical_Examination_Findings", {})
        vs = pe.get("Vital_Signs", {})
        abd = pe.get("Abdominal_Examination", {})
        tests = ctx.get("Test_Results", {})

        bits: List[str] = []
        # 关键线索示例（可按病种扩展）
        if "right lower quadrant" in _normalize(hx):
            bits.append("RLQ pain")
        if "rovsing" in _normalize(json.dumps(abd)):
            bits.append("Rovsing’s sign +")
        if "wbc" in (k := " ".join(tests.get("Complete_Blood_Count", {}).keys()).lower()) \
           or any("wbc" in _normalize(v) for v in tests.get("Complete_Blood_Count", {}).values()):
            # 简化：只要发现 WBC 且有“elevated”字样
            cbc = tests.get("Complete_Blood_Count", {})
            if any("elevated" in _normalize(str(v)) for v in cbc.values()):
                bits.append("WBC elevated")
        if "ultrasound_abdomen" in "".join(tests.get("Imaging", {}).keys()).lower():
            us = tests.get("Imaging", {}).get("Ultrasound_Abdomen", {})
            if "appendix" in _normalize(json.dumps(us)):
                bits.append("US: inflamed appendix")
        # Vital signs 简单提取
        if vs:
            if "38" in _normalize(json.dumps(vs)) or "fever" in _normalize(hx):
                bits.append("fever")

        return ", ".join(bits[:6])

# --------- 将 OSCE JSON 转为 Samples ---------

def _compose_question_from_osce(case: Dict[str, Any]) -> str:
    """
    把结构化 OSCE 字段串联成临床 vignette（最简安全提示）：
    - 只包含给定信息，不诱导“生成超范围内容”
    """
    actor = case.get("Patient_Actor", {})
    demo = actor.get("Demographics", "")
    hx = actor.get("History", "")
    ros = actor.get("Review_of_Systems", "")
    pe = case.get("Physical_Examination_Findings", {})
    vit = pe.get("Vital_Signs", {})
    abd = pe.get("Abdominal_Examination", {})
    tests = case.get("Test_Results", {})

    def _kv(d: Dict[str, Any]) -> str:
        if not d: return ""
        return "; ".join(f"{k}: {v}" for k, v in d.items())

    vignette = []
    vignette.append(f"Demographics: {demo}".strip())
    if hx: vignette.append(f"History: {hx}")
    if ros: vignette.append(f"ROS: {ros}")
    if vit: vignette.append("Vitals: " + _kv(vit))
    if abd: vignette.append("Abd Exam: " + _kv(abd))
    if tests:
        # 仅放关键信息，避免过长
        if "Complete_Blood_Count" in tests:
            vignette.append("CBC: " + _kv(tests["Complete_Blood_Count"]))
        if "Imaging" in tests and "Ultrasound_Abdomen" in tests["Imaging"]:
            uv = tests["Imaging"]["Ultrasound_Abdomen"].get("Findings", "")
            vignette.append(f"Ultrasound: {uv}")

    vignette_txt = "\n".join(vignette)
    question = (
        f"{vignette_txt}\n\n"
        "Task: Based on the above case, provide the single most likely diagnosis.\n"
        "Only output the diagnosis name, without any explanation."
    )
    return question

def make_samples_from_osce(osce_json: str | Dict[str, Any]) -> List[Sample]:
    """
    输入：字符串（JSON）或 dict（与示例相同结构，支持多病例字典）
    输出：可直接喂给 OfflineAdapter.run(...) 的 Sample 列表
    - Sample.question   = 临床 vignette + 任务指令
    - Sample.ground_truth = Correct_Diagnosis
    - Sample.context    = 原始结构化 case（Reflector/Curator 可用）
    """
    if isinstance(osce_json, str):
        data = json.loads(osce_json)
    else:
        data = osce_json

    # 兼容两种包装：{ "OSCE_Examination": {._
