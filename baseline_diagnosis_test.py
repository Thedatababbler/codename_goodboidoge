#!/usr/bin/env python3
"""
Baseline Diagnosis Test

测试 LLM 在不借助 guidebook 和 playbook 的情况下，直接对医疗案例进行诊断的能力。
"""

import argparse
import json
import os
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ace.llm import OpenAIClient
from ace.roles import Retriever
from utils.utils import build_diagnosis_question, extract_ground_truth


# ============================================================
# Baseline Prompt
# ============================================================

BASELINE_PROMPT = """You are a medical diagnosis expert. Based on the following patient case, provide your diagnosis.

{question}

Important: Respond with ONLY the diagnosis name, nothing else. Do not include explanations, reasoning, or any other text.
"""


# ============================================================
# 诊断匹配检查（复制自 offline_medical_adaptation.py）
# ============================================================

def check_diagnosis(prediction: str, ground_truth: str) -> bool:
    """
    诊断匹配检查 - 使用灵活的匹配策略

    支持:
    1. 精确匹配
    2. 包含匹配
    3. 归一化匹配（忽略标点符号差异如 - vs /）
    4. 缩写匹配（如 ADHD = Attention-deficit hyperactivity disorder）
    5. 音调符号处理
    6. 模糊匹配
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
        s = re.sub(r'[-/\\()，,;:：；]', ' ', s)
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


# ============================================================
# 数据加载
# ============================================================

def load_cases(filepath: str) -> List[Dict[str, Any]]:
    """加载 JSONL 格式的病例数据"""
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(data)
    return cases


# ============================================================
# 主测试函数
# ============================================================

def run_baseline_test(
    data_path: str,
    max_cases: Optional[int] = None,
    model: str = "gpt-4o-mini",
    output_path: Optional[str] = None,
    verbose: bool = True,
    temperature: Optional[float] = None,
    filter_by_guideline: bool = False,
    guideline_path: str = "data/guideline_dict.json",
) -> Dict[str, Any]:
    """
    运行 baseline 诊断测试

    Parameters
    ----------
    data_path : str
        病例数据文件路径 (JSONL)
    max_cases : int, optional
        最大测试案例数
    model : str
        使用的模型名称
    output_path : str, optional
        结果输出文件路径
    verbose : bool
        是否打印详细信息
    filter_by_guideline : bool
        是否只测试有 guideline 的案例（与 ACE+Playbook 保持一致）
    guideline_path : str
        guideline 文件路径

    Returns
    -------
    dict
        测试结果统计
    """
    # 加载数据
    cases = load_cases(data_path)
    if max_cases and max_cases < len(cases):
        cases = cases[:max_cases]

    # 如果需要过滤，只保留有 guideline 的案例
    if filter_by_guideline:
        retriever = Retriever(
            json_path=guideline_path,
            source="wikidoc",
            match_fields="title",
        )
        filtered_cases = []
        skipped_diseases = set()
        for case in cases:
            # 提取 ground truth（支持多种数据格式）
            gt = extract_ground_truth(case)
            if gt:
                result = retriever.retrieve_exact(gt)
                if result.entries:
                    filtered_cases.append(case)
                else:
                    skipped_diseases.add(gt)

        if verbose:
            print(f"[FILTER] Original cases: {len(cases)}")
            print(f"[FILTER] Cases with guideline: {len(filtered_cases)}")
            print(f"[FILTER] Skipped diseases ({len(skipped_diseases)}): {sorted(skipped_diseases)[:5]}...")
            print()

        cases = filtered_cases

    total = len(cases)

    if verbose:
        print("=" * 60)
        print("Baseline Diagnosis Test")
        print("=" * 60)
        print(f"Model: {model}")
        print(f"Total Cases: {total}")
        print(f"Data: {data_path}")
        print("=" * 60)
        print()

    # 初始化 LLM 客户端
    if temperature is not None:
        llm = OpenAIClient(model=model, temperature=temperature)
    else:
        llm = OpenAIClient(model=model)

    # 测试结果
    results = []
    correct_count = 0

    # 输出行（用于保存到文件）
    output_lines = [
        "Baseline Diagnosis Test Results",
        "=" * 60,
        f"Model: {model}",
        f"Total Cases: {total}",
        f"Data: {data_path}",
        f"Timestamp: {datetime.now().isoformat()}",
        "=" * 60,
        "",
    ]

    for i, case in enumerate(cases):
        # 提取 ground truth
        ground_truth = extract_ground_truth(case)
        if not ground_truth:
            if verbose:
                print(f"Case {i+1}/{total}: ⚠️  No ground truth, skipping")
            continue

        # 构建诊断问题
        question = build_diagnosis_question(case)

        # 构建完整 prompt
        prompt = BASELINE_PROMPT.format(question=question)

        # 调用 LLM
        try:
            response = llm.complete(prompt)
            prediction = response.text.strip()
        except Exception as e:
            if verbose:
                print(f"Case {i+1}/{total}: ❌ API Error: {e}")
            output_lines.append(f"Case {i+1}/{total}: ❌ API Error: {e}")
            results.append({
                "case_idx": i,
                "ground_truth": ground_truth,
                "prediction": None,
                "is_correct": False,
                "error": str(e),
            })
            continue

        # 检查诊断是否正确
        is_correct = check_diagnosis(prediction, ground_truth)

        if is_correct:
            correct_count += 1

        # 记录结果
        results.append({
            "case_idx": i,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "is_correct": is_correct,
        })

        # 输出进度
        status = "✅" if is_correct else "❌"
        if verbose:
            if is_correct:
                print(f"Case {i+1}/{total}: {status} (GT: {ground_truth})")
            else:
                print(f"Case {i+1}/{total}: {status} (Pred: {prediction}, GT: {ground_truth})")

        output_lines.append(
            f"Case {i+1}/{total}: {status} | Pred: {prediction} | GT: {ground_truth}"
        )

    # 计算准确率
    evaluated = len([r for r in results if r.get("prediction") is not None])
    accuracy = correct_count / evaluated if evaluated > 0 else 0

    # 最终统计
    summary = {
        "total_cases": total,
        "evaluated": evaluated,
        "correct": correct_count,
        "incorrect": evaluated - correct_count,
        "accuracy": accuracy,
        "model": model,
    }

    if verbose:
        print()
        print("=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Evaluated: {evaluated}/{total}")
        print(f"Correct: {correct_count}")
        print(f"Incorrect: {evaluated - correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        print("=" * 60)

    # 保存结果
    output_lines.extend([
        "",
        "=" * 60,
        "Results Summary",
        "=" * 60,
        f"Evaluated: {evaluated}/{total}",
        f"Correct: {correct_count}",
        f"Incorrect: {evaluated - correct_count}",
        f"Accuracy: {accuracy:.2%}",
        "=" * 60,
    ])

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        if verbose:
            print(f"\n[SAVED] Results saved to: {output_path}")

    return {
        "summary": summary,
        "results": results,
    }


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline Diagnosis Test - Test LLM diagnosis without playbook"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to JSONL data file"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to test"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: from OPENAI_MODEL env or gpt-4o-mini)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for LLM (some models like gpt-5-nano only support 1.0)"
    )
    parser.add_argument(
        "--filter-by-guideline",
        action="store_true",
        help="Only test cases that have corresponding guidelines (for fair comparison with ACE+Playbook)"
    )
    parser.add_argument(
        "--guideline-path",
        type=str,
        default="data/guideline_dict.json",
        help="Path to guideline JSON file"
    )

    args = parser.parse_args()

    # 确定模型
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # 默认输出文件
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"baseline_results_{timestamp}.txt"

    # 运行测试
    run_baseline_test(
        data_path=args.data,
        max_cases=args.max_cases,
        model=model,
        output_path=output_path,
        verbose=not args.quiet,
        temperature=args.temperature,
        filter_by_guideline=args.filter_by_guideline,
        guideline_path=args.guideline_path,
    )


if __name__ == "__main__":
    main()
