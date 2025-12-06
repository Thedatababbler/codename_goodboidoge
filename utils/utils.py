import re
from typing import Dict, Any, List, Optional, Union


def build_diagnosis_question(case: Dict[str, Any], *, redact_conclusive: bool = True) -> str:
    """
    将结构化医疗病例字典转换为诊断型问句（仅要求输出诊断名）。
    
    支持的数据格式（兼容 OSCE/AgentClinic 风格）：
    - OSCE_Examination 包装或直接的 case 结构
    - 多层嵌套的 Test_Results（如 Imaging -> Ultrasound_Abdomen -> Findings）
    - 多层嵌套的 Physical_Examination_Findings
    
    Parameters
    ----------
    case : dict
        形如 OSCE/AgentClinic 病例结构（字段名大小写不敏感）。
    redact_conclusive : bool, default=True
        若为 True，自动从"检查/活检"文字中去除直接点名诊断的描述，避免答案剧透。
    
    Returns
    -------
    question : str
        一段 vignette + 任务指令，适合喂给 Generator（只需输出诊断名）。
    """
    
    # ================================================================
    # 工具函数
    # ================================================================
    
    def get_path(d: Dict[str, Any], *keys: str) -> Any:
        """按路径获取嵌套字典中的值，大小写不敏感"""
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return None
            matches = [kk for kk in cur.keys() if kk.lower() == k.lower()]
            if not matches:
                return None
            cur = cur[matches[0]]
        return cur

    def norm_str(x: Optional[str]) -> str:
        """规范化字符串"""
        return str(x).strip() if x is not None else ""
    
    def format_key(key: str) -> str:
        """格式化键名：下划线转空格"""
        return key.replace("_", " ")

    def kvline(d: Optional[Dict[str, Any]], separator: str = "; ") -> str:
        """将字典转为 key: value 形式的单行字符串"""
        if not isinstance(d, dict) or not d:
            return ""
        parts = []
        for k, v in d.items():
            if v not in (None, "", [], {}):
                if isinstance(v, (dict, list)):
                    continue  # 跳过复杂嵌套，留给递归处理
                parts.append(f"{format_key(k)}: {v}")
        return separator.join(parts)

    def maybe_redact_findings(text: str, gold: str) -> str:
        """如果检查结果中包含诊断答案，进行脱敏"""
        if not redact_conclusive or not gold:
            return text
        t = text.lower()
        g = gold.lower().strip()
        grams = [w for w in re.split(r"[\s\-_/]+", g) if w and len(w) > 2]
        # 检查是否有足够的匹配
        hit = (g and g in t) or (len(grams) >= 2 and sum(w in t for w in grams) >= max(2, len(grams) - 1))
        if hit:
            return "[Findings redacted to avoid revealing the diagnosis directly.]"
        return text

    def flatten_nested_dict(
        d: Dict[str, Any], 
        prefix: str = "", 
        gold: str = "",
        depth: int = 0,
        max_depth: int = 3,
    ) -> List[str]:
        """
        递归展平嵌套字典，返回格式化的行列表。
        
        处理场景：
        - Imaging -> Ultrasound_Abdomen -> Findings: "..."
        - Complete_Blood_Count -> WBC: "...", Hemoglobin: "..."
        """
        if depth > max_depth:
            return []
        
        lines = []
        
        for key, value in d.items():
            formatted_key = format_key(key)
            full_key = f"{prefix} - {formatted_key}" if prefix else formatted_key
            
            if value is None or value == "" or value == [] or value == {}:
                continue
            
            if isinstance(value, str):
                # 字符串值：检查是否需要脱敏
                if key.lower() in ("findings", "result", "impression", "conclusion", "report"):
                    value = maybe_redact_findings(value, gold)
                lines.append(f"{full_key}: {value}")
            
            elif isinstance(value, (int, float)):
                lines.append(f"{full_key}: {value}")
            
            elif isinstance(value, list):
                # 列表：转为逗号分隔的字符串
                if all(isinstance(x, str) for x in value):
                    lines.append(f"{full_key}: {', '.join(value)}")
                else:
                    lines.append(f"{full_key}: {value}")
            
            elif isinstance(value, dict):
                # 嵌套字典：递归处理
                # 检查是否是简单的 key-value 字典（无嵌套）
                has_nested = any(isinstance(v, dict) for v in value.values())
                
                if not has_nested:
                    # 简单字典：展平为一行
                    kv = kvline(value)
                    if kv:
                        lines.append(f"{full_key}: {kv}")
                else:
                    # 复杂嵌套：递归展开
                    nested_lines = flatten_nested_dict(
                        value, prefix=full_key, gold=gold, depth=depth + 1
                    )
                    lines.extend(nested_lines)
        
        return lines

    # ================================================================
    # 解析病例结构
    # ================================================================
    
    # 支持 OSCE_Examination 包装
    if "OSCE_Examination" in case:
        case = case["OSCE_Examination"]
    
    # 获取 ground truth（用于脱敏）
    gold = norm_str(get_path(case, "Correct_Diagnosis")) or norm_str(get_path(case, "CorrectDiagnosis"))
    
    # 获取主要字段
    obj_doctor = norm_str(get_path(case, "Objective_for_Doctor"))
    actor = get_path(case, "Patient_Actor") or {}
    
    # 患者基本信息
    demo = norm_str(get_path(actor, "Demographics"))
    hist = norm_str(get_path(actor, "History"))
    ros = norm_str(get_path(actor, "Review_of_Systems"))
    symptoms = get_path(actor, "Symptoms") or {}
    pms = norm_str(get_path(actor, "Past_Medical_History"))
    social = norm_str(get_path(actor, "Social_History"))
    family = norm_str(get_path(actor, "Family_History"))
    allergies = get_path(actor, "Allergies")
    medications = get_path(actor, "Current_Medications") or get_path(actor, "Medications")
    
    # 体格检查
    pe = get_path(case, "Physical_Examination_Findings") or {}
    
    # 检查/检验结果
    tests = get_path(case, "Test_Results") or {}
    
    # ================================================================
    # 组装 vignette
    # ================================================================
    
    lines: List[str] = []
    
    # 1. 基本人口学信息
    if demo:
        lines.append(f"Demographics: {demo}")
    
    # 2. 主诉/病史
    if hist:
        lines.append(f"History: {hist}")
    
    # 3. 症状
    prim = norm_str(get_path(symptoms, "Primary_Symptom"))
    if prim:
        lines.append(f"Primary Symptom: {prim}")
    
    sec = get_path(symptoms, "Secondary_Symptoms")
    if isinstance(sec, list) and sec:
        lines.append("Secondary Symptoms: " + "; ".join(str(s) for s in sec if s))
    elif isinstance(sec, str) and sec:
        lines.append(f"Secondary Symptoms: {sec}")
    
    # 4. 既往史、社会史、家族史
    if pms:
        lines.append(f"Past Medical History: {pms}")
    if social:
        lines.append(f"Social History: {social}")
    if family:
        lines.append(f"Family History: {family}")
    
    # 5. 过敏和用药
    if allergies:
        if isinstance(allergies, list):
            lines.append(f"Allergies: {', '.join(str(a) for a in allergies)}")
        else:
            lines.append(f"Allergies: {allergies}")
    
    if medications:
        if isinstance(medications, list):
            lines.append(f"Current Medications: {', '.join(str(m) for m in medications)}")
        else:
            lines.append(f"Current Medications: {medications}")
    
    # 6. 系统回顾
    if ros:
        lines.append(f"Review of Systems: {ros}")
    
    # 7. 体格检查（递归展平）
    if pe:
        lines.append("")  # 空行分隔
        lines.append("=== Physical Examination ===")
        
        # Vital Signs 单独处理（保持紧凑）
        vitals = get_path(pe, "Vital_Signs")
        if isinstance(vitals, dict) and vitals:
            lines.append("Vital Signs: " + kvline(vitals))
        
        # 其他体格检查项
        for key, value in pe.items():
            if key.lower() == "vital_signs":
                continue
            if isinstance(value, dict) and value:
                formatted_key = format_key(key)
                # 检查是否是简单的 key-value
                has_nested = any(isinstance(v, dict) for v in value.values())
                if not has_nested:
                    kv = kvline(value)
                    if kv:
                        lines.append(f"{formatted_key}: {kv}")
                else:
                    # 复杂嵌套
                    nested = flatten_nested_dict(value, prefix=formatted_key, gold=gold)
                    lines.extend(nested)
    
    # 8. 检查/检验结果（递归展平，支持 Imaging 等多层嵌套）
    if tests:
        lines.append("")  # 空行分隔
        lines.append("=== Test Results ===")
        
        for panel_name, panel_content in tests.items():
            if panel_content is None or panel_content == {} or panel_content == "":
                continue
            
            formatted_panel = format_key(panel_name)
            
            if isinstance(panel_content, str):
                # 直接是字符串结果
                safe_content = maybe_redact_findings(panel_content, gold)
                lines.append(f"{formatted_panel}: {safe_content}")
            
            elif isinstance(panel_content, dict):
                # 检查嵌套深度
                has_nested = any(isinstance(v, dict) for v in panel_content.values())
                
                if not has_nested:
                    # 简单字典：如 Complete_Blood_Count -> WBC, Hemoglobin 等
                    kv = kvline(panel_content)
                    if kv:
                        lines.append(f"{formatted_panel}: {kv}")
                else:
                    # 复杂嵌套：如 Imaging -> Ultrasound_Abdomen -> Findings
                    nested_lines = flatten_nested_dict(
                        panel_content, prefix=formatted_panel, gold=gold
                    )
                    lines.extend(nested_lines)
    
    # ================================================================
    # 组装最终问题
    # ================================================================
    
    vignette = "\n".join(line for line in lines if line is not None).strip()
    
    # 任务指令
    task = "Task: Based on the above case, provide the single most likely diagnosis."
    
    # 题干开场白
    header = obj_doctor or "Objective: Evaluate and diagnose the patient."
    
    question = f"{header}\n\n{vignette}\n\n{task}"
    return question


def extract_ground_truth(case: Dict[str, Any]) -> Optional[str]:
    """
    从病例中提取 ground truth 诊断。
    
    支持多种字段名：
    - Correct_Diagnosis
    - CorrectDiagnosis
    - correct_diagnosis
    - diagnosis
    - Diagnosis
    """
    # 支持 OSCE_Examination 包装
    if "OSCE_Examination" in case:
        case = case["OSCE_Examination"]
    
    # 尝试多种字段名
    for key in ["Correct_Diagnosis", "CorrectDiagnosis", "correct_diagnosis", "Diagnosis", "diagnosis"]:
        if key in case and case[key]:
            return str(case[key]).strip()
    
    return None


def build_case_context(case: Dict[str, Any]) -> str:
    """
    构建病例的上下文描述（不含任务指令）。
    用于 Generator 的 context 参数。
    """
    # 复用 build_diagnosis_question 但去掉任务指令
    full_question = build_diagnosis_question(case, redact_conclusive=True)
    
    # 移除最后的 "Task: ..." 部分
    lines = full_question.split("\n")
    context_lines = []
    for line in lines:
        if line.strip().startswith("Task:"):
            break
        context_lines.append(line)
    
    return "\n".join(context_lines).strip()
