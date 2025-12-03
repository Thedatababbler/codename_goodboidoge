import re
from typing import Dict, Any, List, Optional


def build_diagnosis_question(case: Dict[str, Any], *, redact_conclusive: bool = True) -> str:
    """
    将结构化医疗病例字典转换为诊断型问句（仅要求输出诊断名）。
    
    Parameters
    ----------
    case : dict
        形如你示例中的 OSCE/病例结构（字段名大小写不敏感）。
    redact_conclusive : bool, default=True
        若为 True，自动从“检查/活检”文字中**去除**直接点名诊断的描述，避免把答案剧透进题干。
    
    Returns
    -------
    question : str
        一段 vignette + 任务指令，适合喂给你的 Generator（只需输出诊断名）。
    """
    # 小工具
    def get_path(d: Dict[str, Any], *keys: str) -> Any:
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return None
            # 兼容不同大小写/下划线
            matches = [kk for kk in cur.keys() if kk.lower() == k.lower()]
            if not matches:
                return None
            cur = cur[matches[0]]
        return cur

    def norm_str(x: Optional[str]) -> str:
        return str(x).strip() if x is not None else ""

    def kvline(d: Optional[Dict[str, Any]]) -> str:
        if not isinstance(d, dict) or not d:
            return ""
        return "; ".join(f"{k}: {v}" for k, v in d.items() if v not in (None, ""))

    def maybe_redact_findings(text: str, gold: str) -> str:
        if not redact_conclusive:
            return text
        # 如果检查报告文本里包含/近似包含 gold 诊断词，做保护性脱敏
        t = text.lower()
        g = gold.lower().strip()
        # 允许把 gold 拆成词组后逐词匹配，避免完全相同才触发
        grams = [w for w in re.split(r"[\s\-_/]+", g) if w]
        hit = (g and g in t) or (len(grams) >= 2 and sum(w in t for w in grams) >= max(2, len(grams)-1))
        if hit:
            return "[Findings redacted to avoid revealing the diagnosis directly.]"
        return text

    # 取关键字段
    obj_doctor   = norm_str(get_path(case, "Objective_for_Doctor"))
    actor        = get_path(case, "Patient_Actor") or {}
    demo         = norm_str(get_path(actor, "Demographics"))
    hist         = norm_str(get_path(actor, "History"))
    ros          = norm_str(get_path(actor, "Review_of_Systems"))
    symptoms     = get_path(actor, "Symptoms") or {}
    pms          = norm_str(get_path(actor, "Past_Medical_History"))
    social       = norm_str(get_path(actor, "Social_History"))

    pe           = get_path(case, "Physical_Examination_Findings") or {}
    vitals       = get_path(pe, "Vital_Signs")
    # 可能是“Abdominal_Examination / Oral_Examination / Neuro_Examination”等不同域
    other_pe_keys = [k for k in pe.keys() if k.lower() not in {"vital_signs"}]

    tests        = get_path(case, "Test_Results") or {}
    gold         = norm_str(get_path(case, "Correct_Diagnosis")) or norm_str(get_path(case, "CorrectDiagnosis"))

    # 组装 vignette（按信息重要性排序）
    lines: List[str] = []
    if demo:     lines.append(f"Demographics: {demo}")
    if hist:     lines.append(f"History: {hist}")

    # 症状
    prim = norm_str(get_path(symptoms, "Primary_Symptom"))
    if prim:
        lines.append(f"Primary Symptom: {prim}")
    sec = get_path(symptoms, "Secondary_Symptoms")
    if isinstance(sec, list) and sec:
        lines.append("Secondary Symptoms: " + "; ".join(map(str, sec)))

    if pms:      lines.append(f"Past Medical History: {pms}")
    if social:   lines.append(f"Social History: {social}")
    if ros:      lines.append(f"Review of Systems: {ros}")

    # 体格检查
    if isinstance(vitals, dict) and vitals:
        lines.append("Vital Signs: " + kvline(vitals))
    for k in other_pe_keys:
        sec_pe = get_path(pe, k)
        if isinstance(sec_pe, dict) and sec_pe:
            lines.append(f"{k.replace('_',' ')}: " + kvline(sec_pe))

    # 检查/检验（避免直接剧透）
    if isinstance(tests, dict) and tests:
        for panel, content in tests.items():
            if isinstance(content, dict) and content:
                # 若子项是 {Findings: "..."} 则尝试去泄露
                findings = content.get("Findings") or content.get("findings")
                if isinstance(findings, str):
                    safe_findings = maybe_redact_findings(findings, gold)
                    lines.append(f"{panel.replace('_',' ')} Findings: {safe_findings}")
                else:
                    # 其它键值直接展示
                    lines.append(f"{panel.replace('_',' ')}: " + kvline(content))

    vignette = "\n".join(lines).strip()

    # 任务指令（对齐你 Generator 的用法）
    task = (
        "Task: Based on the above case, provide the single most likely diagnosis.\n"
        # "Only output the diagnosis name, without any explanation."
    )

    # 如果有总目标描述，可以用作题干开场白
    header = obj_doctor or "Objective: Evaluate and diagnose the patient."

    question = f"{header}\n\n{vignette}\n\n{task}"
    return question