"""
医疗领域专用的 Reflector 和 Curator 组件。
支持对 GuidelinePlaybook 的 Delta Update 操作建议和执行。

Delta Update 原则（遵循 ACE 论文）：
- 所有更新都是增量的，不是全量替换
- 操作类型：ADD_BULLET, UPDATE_BULLET, REMOVE_BULLET, TAG_BULLET, ADD_SECTION 等
- 支持 Retrieval Signatures section 用于改善症状→疾病的检索
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .llm import LLMClient, LLMResponse
from .playbook import GuidelinePlaybook, GuidelineBullet, GuidelineSection, DeltaOperation, DeltaBatch
from .roles import GeneratorOutput, _safe_json_loads


# ============================================================================
# 常量定义
# ============================================================================

# Retrieval Signatures Section 的固定ID和标题
RETRIEVAL_SECTION_ID = "sec-retrieval"
RETRIEVAL_SECTION_TITLE = "Retrieval Signatures"

# Exclusion Rules Section 的固定ID和标题（用于存储排除规则）
EXCLUSION_SECTION_ID = "sec-exclusion"
EXCLUSION_SECTION_TITLE = "Exclusion Rules"


# ============================================================================
# 医疗专用 Prompt 模板 - 支持 Retrieval Signatures 和 Delta Update
# ============================================================================

MEDICAL_REFLECTOR_PROMPT = """\
You are a senior medical reviewer analyzing a diagnostic reasoning trajectory.

Your tasks:
1. Evaluate whether the diagnosis reasoning followed clinical guidelines correctly
2. Identify errors or gaps in reasoning
3. Assess which bullets were helpful/harmful/neutral
4. **IMPORTANT**: Propose Delta Updates (incremental changes) to improve the playbook
5. **CRITICAL**: Update the "Retrieval Signatures" section to improve future case retrieval

================ CASE INFORMATION ================
Question/Vignette:
{question}

================ MODEL REASONING ================
Reasoning: {reasoning}
Prediction: {prediction}
Ground Truth: {ground_truth}

================ GUIDELINE PLAYBOOK ================
Title: {playbook_title} | ID: {playbook_id}

Referenced Bullets:
{playbook_excerpt}

Full Structure:
{playbook_structure}

================ FEEDBACK ================
{feedback}

================ ANALYSIS INSTRUCTIONS ================

**STEP 1: Determine if prediction is CORRECT or INCORRECT**

**IF INCORRECT - Perform Deep Error Analysis:**

1. **Error Classification** (choose ONE):
   - MISSED_KEY_SYMPTOM: Critical symptom/finding was present but ignored
   - WRONG_DIFFERENTIAL: Confused with a similar disease
   - INCOMPLETE_REASONING: Reasoning chain was incomplete or flawed
   - KNOWLEDGE_GAP: Lacked necessary medical knowledge

2. **Differential Analysis** (MOST IMPORTANT for learning):
   - Why did the model predict the wrong answer instead of the correct one?
   - What symptoms/findings in THIS case point to the correct answer but NOT the prediction?
   - What are the KEY DIFFERENTIATORS between these two conditions?

3. **Corrective Patterns** (extract 5-8 patterns):
   - Patterns that would PREVENT this specific error in future
   - Each pattern should be generalizable to similar cases
   - Include NEGATIVE patterns (exclusion rules)

**IF CORRECT - Extract Success Patterns:**
1. What key findings led to the correct diagnosis?
2. Extract 2-3 reusable diagnostic patterns

================ RETRIEVAL SIGNATURES ================
Section ID: "{retrieval_section_id}"

This special section stores clinical patterns for case-to-guideline matching.
Each bullet should capture ONE distinct clinical pattern:
- Key symptom combinations
- Risk factor profiles
- Lab/imaging patterns
- Differential diagnosis cues

When case reveals NEW patterns not already present, ADD them.
When existing patterns are MISLEADING, TAG as harmful or UPDATE.

================ OUTPUT FORMAT ================
Return a SINGLE valid JSON object:
{{
  "is_correct": true or false,
  "reasoning": "<your detailed analysis>",

  "error_type": "MISSED_KEY_SYMPTOM|WRONG_DIFFERENTIAL|INCOMPLETE_REASONING|KNOWLEDGE_GAP|null",
  "differential_analysis": {{
    "predicted": "<model's answer or null if correct>",
    "correct": "<ground truth>",
    "key_differentiators": ["<finding that distinguishes correct from predicted>"],
    "missed_findings": ["<critical finding that was ignored>"]
  }},

  "error_identification": "<specific errors or 'None'>",
  "root_cause_analysis": "<why errors occurred or 'N/A'>",
  "correct_approach": "<correct diagnostic approach>",
  "key_insight": "<reusable clinical takeaway>",

  "retrieval_patterns": [
    "<pattern 1>",
    "<pattern 2>"
  ],

  "exclusion_rules": [
    "If [finding] present, rule OUT [disease] because [reason]"
  ],

  "bullet_tags": [
    {{"id": "<bullet-id>", "tag": "helpful|harmful|neutral"}}
  ],

  "proposed_operations": [
    {{
      "type": "ADD_BULLET|UPDATE_BULLET|TAG_BULLET|REMOVE_BULLET",
      "section_id": "<target section id>",
      "bullet_id": "<for UPDATE/REMOVE/TAG>",
      "content": "<bullet content for ADD/UPDATE>",
      "metadata": {{"helpful": 0, "harmful": 0}},
      "note": "<reason for this delta>"
    }}
  ]
}}

IMPORTANT RULES:
1. For INCORRECT cases:
   - Focus on differential analysis - understand WHY the error happened
   - Extract 5-8 retrieval_patterns that would help prevent this error
   - Generate exclusion_rules based on differential analysis
2. For CORRECT cases:
   - Extract 2-3 retrieval_patterns that capture the successful reasoning
   - exclusion_rules can be empty
3. retrieval_patterns will be auto-converted to ADD_BULLET for Retrieval Signatures section
4. exclusion_rules will be auto-converted to ADD_BULLET for Exclusion Rules section
5. Only add patterns NOT already present in the playbook
6. Make patterns generalizable - avoid case-specific details, focus on reusable clinical logic

Now analyze and produce the JSON:
"""


MEDICAL_CURATOR_PROMPT = """\
You are the curator of a medical guideline playbook applying DELTA UPDATES.

Your job:
1. Validate proposed operations from the reflector
2. **ENSURE INCREMENTAL UPDATES** - never replace entire sections
3. **CHECK FOR DUPLICATES** - skip operations that add duplicate content
4. Merge the retrieval patterns into the Retrieval Signatures section
5. Apply TAG operations to track bullet usefulness over time

================ CURRENT PLAYBOOK STATE ================
Title: {playbook_title} | ID: {playbook_id}
Stats: {stats}

Current Structure (with Retrieval Signatures):
{playbook}

================ REFLECTION ANALYSIS ================
{reflection}

================ CONTEXT ================
Question: {question_context}
Progress: {progress}

================ DELTA UPDATE RULES ================
1. **RETRIEVAL SIGNATURES** (section_id: "{retrieval_section_id}"):
   - Convert each retrieval_pattern to an ADD_BULLET operation
   - Skip patterns that duplicate existing bullets (check similarity)
   - Each pattern becomes one concise bullet for retrieval

2. **DEDUPLICATION**:
   - Before adding a bullet, check if similar content exists
   - If similar exists, consider UPDATE instead of ADD, or skip

3. **COUNTER UPDATES**:
   - Always include TAG operations to update helpful/harmful counters
   - This tracks which bullets are useful across multiple cases

================ OUTPUT FORMAT ================
Return a SINGLE valid JSON object:
{{
  "reasoning": "<your curation rationale>",
  "operations": [
    {{
      "type": "ADD_BULLET|UPDATE_BULLET|TAG_BULLET|REMOVE_BULLET",
      "section_id": "<section id>",
      "bullet_id": "<for UPDATE/TAG/REMOVE>",
      "content": "<bullet content>",
      "metadata": {{"helpful": 1, "harmful": 0}}
    }}
  ]
}}

If no valid operations, return empty operations array.
Now produce the JSON:
"""


def _format_optional(value: Optional[str]) -> str:
    return value if value else "(none)"


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class BulletTag:
    """单个bullet的标签（helpful/harmful/neutral）"""
    id: str
    tag: str  # "helpful", "harmful", "neutral"


@dataclass
class ProposedOperation:
    """Reflector提议的单个CRUD操作（Delta Update）"""
    type: str  # ADD_BULLET, UPDATE_BULLET, REMOVE_BULLET, TAG_BULLET, etc.
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    bullet_id: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    note: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposedOperation":
        return cls(
            type=str(data.get("type", "")),
            section_id=data.get("section_id"),
            section_title=data.get("section_title"),
            bullet_id=data.get("bullet_id"),
            content=data.get("content"),
            metadata=data.get("metadata", {}),
            note=data.get("note"),
        )

    def to_delta_operation(self) -> DeltaOperation:
        """转换为playbook.py中定义的DeltaOperation"""
        return DeltaOperation(
            type=self.type,
            section_id=self.section_id,
            section_title=self.section_title,
            bullet_id=self.bullet_id,
            content=self.content,
            metadata=self.metadata,
            note=self.note,
        )


@dataclass
class DifferentialAnalysis:
    """差异性分析结构（用于错误案例分析）"""
    predicted: Optional[str] = None
    correct: Optional[str] = None
    key_differentiators: List[str] = field(default_factory=list)
    missed_findings: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["DifferentialAnalysis"]:
        if not data:
            return None
        return cls(
            predicted=data.get("predicted"),
            correct=data.get("correct"),
            key_differentiators=data.get("key_differentiators", []),
            missed_findings=data.get("missed_findings", []),
        )


@dataclass
class MedicalReflectorOutput:
    """MedicalReflector的输出结构"""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    retrieval_patterns: List[str]  # 检索模式
    bullet_tags: List[BulletTag]
    proposed_operations: List[ProposedOperation]
    raw: Dict[str, Any]
    # 新增字段
    is_correct: bool = True  # 预测是否正确
    error_type: Optional[str] = None  # 错误类型分类
    differential_analysis: Optional[DifferentialAnalysis] = None  # 差异性分析
    exclusion_rules: List[str] = field(default_factory=list)  # 排除规则

    def to_delta_batch(self, include_retrieval_ops: bool = True) -> DeltaBatch:
        """
        将proposed_operations转换为DeltaBatch。
        
        Parameters
        ----------
        include_retrieval_ops : bool
            是否自动将retrieval_patterns转换为ADD_BULLET操作
        """
        ops = [op.to_delta_operation() for op in self.proposed_operations]
        
        # 将retrieval_patterns转换为对Retrieval Signatures section的ADD_BULLET操作
        if include_retrieval_ops and self.retrieval_patterns:
            for pattern in self.retrieval_patterns:
                if pattern.strip():
                    ops.append(DeltaOperation(
                        type="ADD_BULLET",
                        section_id=RETRIEVAL_SECTION_ID,
                        content=pattern.strip(),
                        metadata={"helpful": 1},
                        note="Auto-generated from case retrieval pattern",
                    ))
        
        return DeltaBatch(operations=ops, created_by="medical-reflector")


# ============================================================================
# Playbook 辅助函数
# ============================================================================

def ensure_retrieval_section(playbook: GuidelinePlaybook) -> GuidelineSection:
    """
    确保playbook中存在Retrieval Signatures section。
    如果不存在则创建。
    
    Returns
    -------
    GuidelineSection
        Retrieval Signatures section
    """
    section = playbook.get_section(RETRIEVAL_SECTION_ID)
    if section is None:
        section = playbook.add_section(
            title=RETRIEVAL_SECTION_TITLE,
            section_id=RETRIEVAL_SECTION_ID,
        )
        # 添加一个初始bullet说明这个section的用途
        playbook.add_bullet(
            section_id=RETRIEVAL_SECTION_ID,
            content=f"[META] This section contains retrieval signatures - clinical patterns that help identify when this guideline ({playbook.title}) is relevant to a case.",
            metadata={"helpful": 0, "harmful": 0},
        )
    return section


def ensure_exclusion_section(playbook: GuidelinePlaybook) -> GuidelineSection:
    """
    确保playbook中存在Exclusion Rules section。
    如果不存在则创建。

    Returns
    -------
    GuidelineSection
        Exclusion Rules section
    """
    section = playbook.get_section(EXCLUSION_SECTION_ID)
    if section is None:
        section = playbook.add_section(
            title=EXCLUSION_SECTION_TITLE,
            section_id=EXCLUSION_SECTION_ID,
        )
        # 添加一个初始bullet说明这个section的用途
        playbook.add_bullet(
            section_id=EXCLUSION_SECTION_ID,
            content=f"[META] This section contains exclusion rules - patterns that help rule OUT this condition ({playbook.title}) when certain findings are present.",
            metadata={"helpful": 0, "harmful": 0},
        )
    return section


def get_retrieval_signatures(playbook: GuidelinePlaybook) -> List[str]:
    """获取playbook中所有的retrieval signatures"""
    section = playbook.get_section(RETRIEVAL_SECTION_ID)
    if section is None:
        return []

    signatures = []
    for bullet_id in section.bullet_ids:
        bullet = playbook.get_bullet(bullet_id)
        if bullet and not bullet.content.startswith("[META]"):
            signatures.append(bullet.content)
    return signatures


def get_exclusion_rules(playbook: GuidelinePlaybook) -> List[str]:
    """获取playbook中所有的exclusion rules"""
    section = playbook.get_section(EXCLUSION_SECTION_ID)
    if section is None:
        return []

    rules = []
    for bullet_id in section.bullet_ids:
        bullet = playbook.get_bullet(bullet_id)
        if bullet and not bullet.content.startswith("[META]"):
            rules.append(bullet.content)
    return rules


def check_duplicate_exclusion_rule(
    playbook: GuidelinePlaybook,
    new_rule: str,
) -> bool:
    """
    检查新的exclusion rule是否与已有的重复。
    """
    existing = get_exclusion_rules(playbook)
    new_lower = new_rule.lower().strip()

    for rule in existing:
        rule_lower = rule.lower().strip()
        if new_lower in rule_lower or rule_lower in new_lower:
            return True
    return False


def check_duplicate_signature(
    playbook: GuidelinePlaybook, 
    new_pattern: str,
    similarity_threshold: float = 0.8,
) -> bool:
    """
    检查新的retrieval pattern是否与已有的重复。
    
    简单实现：基于字符串包含关系检查。
    可以扩展为使用embedding相似度。
    """
    existing = get_retrieval_signatures(playbook)
    new_lower = new_pattern.lower().strip()
    
    for sig in existing:
        sig_lower = sig.lower().strip()
        # 简单的重复检测：包含关系或高度相似
        if new_lower in sig_lower or sig_lower in new_lower:
            return True
        # 可以添加更复杂的相似度检测
    return False


# ============================================================================
# MedicalReflector - 医疗诊断反思Agent
# ============================================================================

class MedicalReflector:
    """
    医疗领域专用的 Reflector，用于：
    1. 分析Generator的诊断推理过程
    2. 对比预测结果和ground truth
    3. 识别错误原因和改进点
    4. 提出Delta Update建议（增量更新）
    5. 提取Retrieval Patterns用于改善检索
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = MEDICAL_REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: GuidelinePlaybook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        max_refinement_rounds: int = 1,
        **kwargs: Any,
    ) -> MedicalReflectorOutput:
        """
        对Generator的输出进行反思分析，提出Delta Update建议。
        """
        # 确保Retrieval Signatures section和Exclusion Rules section存在
        ensure_retrieval_section(playbook)
        ensure_exclusion_section(playbook)
        
        # 构建playbook摘要
        playbook_excerpt = self._make_playbook_excerpt(playbook, generator_output.bullet_ids)
        playbook_structure = self._make_playbook_structure(playbook)
        
        base_prompt = self.prompt_template.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            playbook_title=playbook.title,
            playbook_id=playbook.guideline_id,
            playbook_excerpt=playbook_excerpt or "(no bullets referenced)",
            playbook_structure=playbook_structure,
            retrieval_section_id=RETRIEVAL_SECTION_ID,
        )
        
        result: Optional[MedicalReflectorOutput] = None
        last_error: Optional[Exception] = None
        
        for round_idx in range(max_refinement_rounds):
            prompt = base_prompt
            for attempt in range(self.max_retries):
                response = self.llm.complete(prompt, **kwargs)
                try:
                    data = _safe_json_loads(response.text)
                    
                    # 解析bullet_tags
                    bullet_tags: List[BulletTag] = []
                    tags_payload = data.get("bullet_tags", [])
                    if isinstance(tags_payload, Sequence):
                        for item in tags_payload:
                            if isinstance(item, dict) and "id" in item and "tag" in item:
                                bullet_tags.append(
                                    BulletTag(
                                        id=str(item["id"]),
                                        tag=str(item["tag"]).lower()
                                    )
                                )
                    
                    # 解析retrieval_patterns（新增）
                    retrieval_patterns: List[str] = []
                    patterns_payload = data.get("retrieval_patterns", [])
                    if isinstance(patterns_payload, list):
                        for p in patterns_payload:
                            if isinstance(p, str) and p.strip():
                                retrieval_patterns.append(p.strip())
                    
                    # 解析proposed_operations
                    proposed_ops: List[ProposedOperation] = []
                    ops_payload = data.get("proposed_operations", [])
                    if isinstance(ops_payload, Sequence):
                        for item in ops_payload:
                            if isinstance(item, dict) and "type" in item:
                                proposed_ops.append(ProposedOperation.from_dict(item))

                    # 解析新增字段：is_correct, error_type, differential_analysis, exclusion_rules
                    is_correct = data.get("is_correct", True)
                    if isinstance(is_correct, str):
                        is_correct = is_correct.lower() == "true"

                    error_type = data.get("error_type")
                    if error_type and error_type.lower() == "null":
                        error_type = None

                    differential_analysis = DifferentialAnalysis.from_dict(
                        data.get("differential_analysis")
                    )

                    exclusion_rules: List[str] = []
                    exclusion_payload = data.get("exclusion_rules", [])
                    if isinstance(exclusion_payload, list):
                        for rule in exclusion_payload:
                            if isinstance(rule, str) and rule.strip():
                                exclusion_rules.append(rule.strip())

                    candidate = MedicalReflectorOutput(
                        reasoning=str(data.get("reasoning", "")),
                        error_identification=str(data.get("error_identification", "")),
                        root_cause_analysis=str(data.get("root_cause_analysis", "")),
                        correct_approach=str(data.get("correct_approach", "")),
                        key_insight=str(data.get("key_insight", "")),
                        retrieval_patterns=retrieval_patterns,
                        bullet_tags=bullet_tags,
                        proposed_operations=proposed_ops,
                        raw=data,
                        is_correct=is_correct,
                        error_type=error_type,
                        differential_analysis=differential_analysis,
                        exclusion_rules=exclusion_rules,
                    )
                    result = candidate

                    # 如果有可操作的输出，提前返回
                    if bullet_tags or proposed_ops or retrieval_patterns or exclusion_rules or candidate.key_insight:
                        return candidate
                    break
                    
                except ValueError as err:
                    last_error = err
                    if attempt + 1 >= self.max_retries:
                        break
                    prompt = (
                        base_prompt
                        + "\n\n请严格输出有效JSON，对双引号进行转义，"
                        "不要输出额外解释性文本。JSON必须以 { 开始，以 } 结束。"
                    )
        
        if result is None:
            raise RuntimeError("MedicalReflector failed to produce a result.") from last_error
        return result

    def _make_playbook_excerpt(
        self, 
        playbook: GuidelinePlaybook, 
        bullet_ids: Sequence[str]
    ) -> str:
        """从playbook中提取被引用的bullet摘要"""
        lines: List[str] = []
        seen = set()
        for bullet_id in bullet_ids:
            if bullet_id in seen:
                continue
            bullet = playbook.get_bullet(bullet_id)
            if bullet:
                seen.add(bullet_id)
                lines.append(f"[{bullet.id}] (section: {bullet.section_id}) {bullet.content[:200]}...")
        return "\n".join(lines) if lines else "(no bullets referenced)"

    def _make_playbook_structure(self, playbook: GuidelinePlaybook) -> str:
        """生成playbook的结构摘要供LLM参考，特别标注Retrieval Signatures section"""
        lines: List[str] = []
        for section in playbook.sections():
            # 特别标注Retrieval Signatures section
            if section.id == RETRIEVAL_SECTION_ID:
                lines.append(f"Section [{section.id}]: {section.title} [RETRIEVAL - for case matching]")
            else:
                lines.append(f"Section [{section.id}]: {section.title}")
            
            for bullet_id in section.bullet_ids[:5]:
                bullet = playbook.get_bullet(bullet_id)
                if bullet:
                    content_preview = bullet.content[:100].replace("\n", " ")
                    tags_str = f"(h={bullet.tags.get('helpful', 0)}, harm={bullet.tags.get('harmful', 0)})"
                    lines.append(f"  - [{bullet.id}] {content_preview}... {tags_str}")
            if len(section.bullet_ids) > 5:
                lines.append(f"  ... and {len(section.bullet_ids) - 5} more bullets")
        return "\n".join(lines) if lines else "(empty playbook)"


# ============================================================================
# MedicalCurator - 医疗Guideline策展Agent
# ============================================================================

@dataclass
class MedicalCuratorOutput:
    """MedicalCurator的输出"""
    reasoning: str
    delta: DeltaBatch
    raw: Dict[str, Any]
    applied_ops: List[str] = field(default_factory=list)
    skipped_ops: List[str] = field(default_factory=list)


class MedicalCurator:
    """
    医疗领域专用的 Curator，用于：
    1. 验证Reflector提出的Delta Update操作
    2. 过滤重复的retrieval patterns
    3. 执行增量更新到playbook
    4. 维护helpful/harmful计数器
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = MEDICAL_CURATOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def curate(
        self,
        *,
        reflection: MedicalReflectorOutput,
        playbook: GuidelinePlaybook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> MedicalCuratorOutput:
        """
        基于Reflector的分析，决定最终的Delta Update操作。
        使用LLM进行验证和去重。
        """
        stats = {
            "sections": len(playbook.sections()),
            "bullets": len(playbook.bullets()),
            "retrieval_signatures": len(get_retrieval_signatures(playbook)),
        }
        
        base_prompt = self.prompt_template.format(
            playbook_title=playbook.title,
            playbook_id=playbook.guideline_id,
            stats=json.dumps(stats),
            playbook=playbook.to_hierarchical_json(include_timestamps=False)[:4000],
            reflection=json.dumps(reflection.raw, ensure_ascii=False, indent=2),
            question_context=question_context,
            progress=progress,
            retrieval_section_id=RETRIEVAL_SECTION_ID,
        )
        
        prompt = base_prompt
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                
                operations: List[DeltaOperation] = []
                ops_payload = data.get("operations", [])
                if isinstance(ops_payload, Sequence):
                    for item in ops_payload:
                        if isinstance(item, dict) and "type" in item:
                            operations.append(DeltaOperation(
                                type=str(item.get("type", "")),
                                section_id=item.get("section_id"),
                                section_title=item.get("section_title"),
                                bullet_id=item.get("bullet_id"),
                                content=item.get("content"),
                                metadata=item.get("metadata", {}),
                                note=item.get("note"),
                            ))
                
                delta = DeltaBatch(
                    operations=operations,
                    created_by="medical-curator",
                )
                
                return MedicalCuratorOutput(
                    reasoning=str(data.get("reasoning", "")),
                    delta=delta,
                    raw=data,
                )
                
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                prompt = (
                    base_prompt
                    + "\n\n提醒：仅输出有效JSON，所有字符串请转义双引号，"
                    "不要添加额外文本。"
                )
        
        raise RuntimeError("MedicalCurator failed to produce valid JSON.") from last_error

    def curate_direct(
        self,
        reflection: MedicalReflectorOutput,
        playbook: GuidelinePlaybook,
        *,
        auto_apply: bool = False,
        deduplicate_retrieval: bool = True,
        deduplicate_exclusion: bool = True,
    ) -> MedicalCuratorOutput:
        """
        直接使用Reflector的建议（不经过LLM二次验证）。

        Parameters
        ----------
        reflection : MedicalReflectorOutput
            Reflector的分析结果
        playbook : GuidelinePlaybook
            当前playbook
        auto_apply : bool
            是否自动应用到playbook
        deduplicate_retrieval : bool
            是否对retrieval patterns进行去重
        deduplicate_exclusion : bool
            是否对exclusion rules进行去重
        """
        # 确保Retrieval Signatures section和Exclusion Rules section存在
        ensure_retrieval_section(playbook)
        ensure_exclusion_section(playbook)

        # 构建delta batch
        all_ops: List[DeltaOperation] = []
        skipped: List[str] = []

        # 1. 添加Reflector提出的操作
        for op in reflection.proposed_operations:
            all_ops.append(op.to_delta_operation())

        # 2. 处理retrieval patterns（去重）
        for pattern in reflection.retrieval_patterns:
            if not pattern.strip():
                continue

            if deduplicate_retrieval and check_duplicate_signature(playbook, pattern):
                skipped.append(f"Duplicate retrieval pattern: {pattern[:50]}...")
                continue

            all_ops.append(DeltaOperation(
                type="ADD_BULLET",
                section_id=RETRIEVAL_SECTION_ID,
                content=pattern.strip(),
                metadata={"helpful": 1},
                note="Retrieval pattern from case",
            ))

        # 3. 处理exclusion rules（去重）- 新增
        for rule in reflection.exclusion_rules:
            if not rule.strip():
                continue

            if deduplicate_exclusion and check_duplicate_exclusion_rule(playbook, rule):
                skipped.append(f"Duplicate exclusion rule: {rule[:50]}...")
                continue

            all_ops.append(DeltaOperation(
                type="ADD_BULLET",
                section_id=EXCLUSION_SECTION_ID,
                content=rule.strip(),
                metadata={"helpful": 1, "source": "error_analysis"},
                note=f"Exclusion rule from error case (error_type: {reflection.error_type})",
            ))

        # 4. 将bullet_tags转换为TAG_BULLET操作
        for tag in reflection.bullet_tags:
            all_ops.append(DeltaOperation(
                type="TAG_BULLET",
                bullet_id=tag.id,
                metadata={tag.tag: 1},
                note=f"Tag from reflection: {tag.tag}",
            ))

        delta = DeltaBatch(operations=all_ops, created_by="medical-curator-direct")

        applied: List[str] = []
        if auto_apply:
            result = playbook.apply_delta(delta)
            applied = result.get("applied", [])
            skipped.extend(result.get("skipped", []))

        return MedicalCuratorOutput(
            reasoning="Direct application of reflector proposals with deduplication",
            delta=delta,
            raw={
                "source": "direct",
                "proposals_count": len(reflection.proposed_operations),
                "retrieval_patterns_count": len(reflection.retrieval_patterns),
                "exclusion_rules_count": len(reflection.exclusion_rules),
                "bullet_tags_count": len(reflection.bullet_tags),
                "is_correct": reflection.is_correct,
                "error_type": reflection.error_type,
            },
            applied_ops=applied,
            skipped_ops=skipped,
        )


# ============================================================================
# 便捷函数
# ============================================================================

def apply_bullet_tags(
    playbook: GuidelinePlaybook, 
    tags: List[BulletTag]
) -> Dict[str, List[str]]:
    """
    将Reflector输出的bullet_tags应用到playbook。
    这是一种Delta Update操作。
    """
    applied = []
    skipped = []
    
    for tag in tags:
        bullet = playbook.get_bullet(tag.id)
        if bullet:
            try:
                bullet.tag(tag.tag, increment=1)
                applied.append(f"{tag.id}: +1 {tag.tag}")
            except Exception as e:
                skipped.append(f"{tag.id}: {str(e)}")
        else:
            skipped.append(f"{tag.id}: bullet not found")
    
    return {"applied": applied, "skipped": skipped}


def create_medical_agents(
    llm: LLMClient,
    *,
    reflector_prompt: Optional[str] = None,
    curator_prompt: Optional[str] = None,
) -> tuple:
    """
    便捷函数：创建配套的MedicalReflector和MedicalCurator。
    """
    reflector = MedicalReflector(
        llm, 
        prompt_template=reflector_prompt or MEDICAL_REFLECTOR_PROMPT
    )
    curator = MedicalCurator(
        llm,
        prompt_template=curator_prompt or MEDICAL_CURATOR_PROMPT
    )
    return reflector, curator


def initialize_playbook_for_medical(
    playbook: GuidelinePlaybook,
    disease_name: str,
) -> GuidelinePlaybook:
    """
    为医疗任务初始化playbook，添加Retrieval Signatures section。
    
    Parameters
    ----------
    playbook : GuidelinePlaybook
        原始playbook
    disease_name : str
        疾病名称
    
    Returns
    -------
    GuidelinePlaybook
        初始化后的playbook
    """
    ensure_retrieval_section(playbook)
    
    # 添加疾病名称作为基础retrieval signature
    playbook.add_bullet(
        section_id=RETRIEVAL_SECTION_ID,
        content=f"Disease: {disease_name}",
        metadata={"helpful": 1},
    )
    
    return playbook
