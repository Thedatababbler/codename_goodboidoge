"""
治疗/给药类任务专用的 Prompt 模板。

适用场景：
- 治疗方案制定 (Treatment Planning)
- 药物选择与给药 (Medication Selection & Dosing)
- 治疗步骤/流程 (Treatment Workflow)

与诊断任务的区别：
- 已知疾病诊断，关注治疗方案的正确性
- 不需要更新 Retrieval Signatures（检索较容易）
- 关注药物相互作用、禁忌症、剂量等
"""

# ============================================================================
# Generator Prompts - 治疗/给药任务
# ============================================================================

TREATMENT_GENERATOR_PROMPT = """\
You are a clinical reasoning agent working on treatment planning tasks.
Given a patient case with an established diagnosis, provide an appropriate treatment plan.

================ PLAYBOOK (Clinical Guidelines) ================
{playbook}
================================================================

================ PATIENT CASE ================
{context}
===============================================

================ TASK ================
{question}
======================================

REQUIREMENTS:
1) Return EXACTLY ONE valid JSON object.
2) JSON keys:
   - "reasoning": Brief clinical rationale for your treatment choices (2-3 sentences).
   - "treatment_plan": Your recommended treatment (medications, procedures, lifestyle, etc.)
   - "key_considerations": Important factors considered (contraindications, drug interactions, patient factors).
   - "bullet_ids": Array of playbook bullet IDs you referenced; [] if none.

OUTPUT JSON:
{{
  "reasoning": "<clinical rationale>",
  "treatment_plan": "<detailed treatment recommendation>",
  "key_considerations": "<important factors>",
  "bullet_ids": ["<id1>", "<id2>"]
}}
"""


MEDICATION_GENERATOR_PROMPT = """\
You are a clinical pharmacology agent assisting with medication selection and dosing.
Given a patient case with diagnosis and relevant information, recommend appropriate medication(s).

================ PLAYBOOK (Drug Guidelines & Protocols) ================
{playbook}
========================================================================

================ PATIENT INFORMATION ================
{context}
=====================================================

================ MEDICATION QUERY ================
{question}
==================================================

REQUIREMENTS:
1) Return EXACTLY ONE valid JSON object.
2) JSON keys:
   - "reasoning": Pharmacological rationale for medication selection.
   - "medication": Primary medication recommendation with dosing.
   - "alternatives": Alternative medications if applicable.
   - "contraindications_checked": Relevant contraindications/interactions considered.
   - "monitoring": Recommended monitoring parameters.
   - "bullet_ids": Array of playbook bullet IDs referenced.

OUTPUT JSON:
{{
  "reasoning": "<pharmacological rationale>",
  "medication": {{
    "name": "<drug name>",
    "dose": "<dose>",
    "route": "<route>",
    "frequency": "<frequency>",
    "duration": "<duration if applicable>"
  }},
  "alternatives": ["<alternative 1>", "<alternative 2>"],
  "contraindications_checked": ["<checked item 1>", "<checked item 2>"],
  "monitoring": ["<parameter 1>", "<parameter 2>"],
  "bullet_ids": []
}}
"""


WORKFLOW_GENERATOR_PROMPT = """\
You are a clinical workflow agent helping to plan treatment steps and procedures.
Given a patient case, provide a structured treatment workflow or procedure plan.

================ PLAYBOOK (Clinical Protocols) ================
{playbook}
===============================================================

================ CASE INFORMATION ================
{context}
==================================================

================ WORKFLOW QUERY ================
{question}
=================================================

REQUIREMENTS:
1) Return EXACTLY ONE valid JSON object.
2) JSON keys:
   - "reasoning": Clinical rationale for the workflow.
   - "workflow_steps": Ordered list of treatment steps.
   - "precautions": Safety precautions and considerations.
   - "expected_outcomes": Expected outcomes and milestones.
   - "bullet_ids": Referenced playbook bullet IDs.

OUTPUT JSON:
{{
  "reasoning": "<clinical rationale>",
  "workflow_steps": [
    {{"step": 1, "action": "<action>", "details": "<details>"}},
    {{"step": 2, "action": "<action>", "details": "<details>"}}
  ],
  "precautions": ["<precaution 1>", "<precaution 2>"],
  "expected_outcomes": ["<outcome 1>", "<outcome 2>"],
  "bullet_ids": []
}}
"""


# ============================================================================
# Reflector Prompts - 治疗/给药任务
# ============================================================================

TREATMENT_REFLECTOR_PROMPT = """\
You are a senior clinical reviewer analyzing a treatment planning trajectory.

Your tasks:
1. Evaluate whether the treatment plan follows clinical guidelines
2. Identify any errors, omissions, or safety concerns
3. Assess which guideline bullets were helpful/harmful
4. Propose Delta Updates to improve the treatment guideline playbook

================ CASE INFORMATION ================
Case: {question}
Diagnosis: {diagnosis}

================ MODEL OUTPUT ================
Treatment Plan: {treatment_plan}
Reasoning: {reasoning}
Key Considerations: {key_considerations}

================ GROUND TRUTH (if available) ================
{ground_truth}

================ GUIDELINE PLAYBOOK ================
Title: {playbook_title} | ID: {playbook_id}
Referenced Bullets: {playbook_excerpt}
Structure: {playbook_structure}

================ FEEDBACK ================
{feedback}

================ YOUR ANALYSIS ================
Return a SINGLE valid JSON object:
{{
  "reasoning": "<your analysis of the treatment plan>",
  "error_identification": "<specific errors or 'None'>",
  "safety_concerns": "<any safety issues identified>",
  "guideline_adherence": "<how well it follows guidelines>",
  "key_insight": "<reusable clinical insight>",
  "bullet_tags": [
    {{"id": "<bullet-id>", "tag": "helpful|harmful|neutral"}}
  ],
  "proposed_operations": [
    {{
      "type": "ADD_BULLET|UPDATE_BULLET|TAG_BULLET|REMOVE_BULLET",
      "section_id": "<section>",
      "bullet_id": "<id if applicable>",
      "content": "<content if ADD/UPDATE>",
      "metadata": {{"helpful": 0, "harmful": 0}},
      "note": "<reason>"
    }}
  ]
}}
"""


MEDICATION_REFLECTOR_PROMPT = """\
You are a clinical pharmacology reviewer analyzing a medication recommendation.

Your tasks:
1. Verify medication selection is appropriate for the indication
2. Check dosing accuracy and safety
3. Identify potential drug interactions or contraindications missed
4. Assess guideline adherence and propose improvements

================ PATIENT CASE ================
{question}
Diagnosis: {diagnosis}

================ MEDICATION RECOMMENDATION ================
Medication: {medication}
Reasoning: {reasoning}
Alternatives: {alternatives}
Contraindications Checked: {contraindications_checked}
Monitoring: {monitoring}

================ GROUND TRUTH (if available) ================
{ground_truth}

================ DRUG GUIDELINE PLAYBOOK ================
Title: {playbook_title} | ID: {playbook_id}
Referenced: {playbook_excerpt}
Structure: {playbook_structure}

================ FEEDBACK ================
{feedback}

================ YOUR ANALYSIS ================
Return a SINGLE valid JSON object:
{{
  "reasoning": "<pharmacological analysis>",
  "dosing_assessment": "<is dosing appropriate?>",
  "interaction_review": "<drug interactions identified>",
  "contraindication_review": "<contraindications assessment>",
  "error_identification": "<errors or 'None'>",
  "key_insight": "<reusable pharmacological insight>",
  "bullet_tags": [{{"id": "<id>", "tag": "helpful|harmful|neutral"}}],
  "proposed_operations": [
    {{
      "type": "ADD_BULLET|UPDATE_BULLET|TAG_BULLET",
      "section_id": "<section>",
      "bullet_id": "<id>",
      "content": "<content>",
      "metadata": {{}},
      "note": "<reason>"
    }}
  ]
}}
"""


# ============================================================================
# Curator Prompt - 通用治疗任务
# ============================================================================

TREATMENT_CURATOR_PROMPT = """\
You are the curator of a treatment guideline playbook applying Delta Updates.

Your job:
1. Validate proposed operations from the reflector
2. Ensure updates are clinically accurate and safe
3. Check for duplicates before adding new bullets
4. Apply incremental updates only

================ CURRENT PLAYBOOK ================
Title: {playbook_title} | ID: {playbook_id}
Stats: {stats}
Structure: {playbook}

================ REFLECTION ANALYSIS ================
{reflection}

================ CONTEXT ================
Case: {question_context}
Progress: {progress}

================ OUTPUT ================
Return a SINGLE valid JSON object:
{{
  "reasoning": "<curation rationale>",
  "operations": [
    {{
      "type": "ADD_BULLET|UPDATE_BULLET|TAG_BULLET|REMOVE_BULLET",
      "section_id": "<section>",
      "bullet_id": "<id>",
      "content": "<content>",
      "metadata": {{"helpful": 1, "harmful": 0}}
    }}
  ]
}}
"""


# ============================================================================
# 任务类型枚举
# ============================================================================

class TreatmentTaskType:
    """治疗任务类型"""
    TREATMENT_PLAN = "treatment_plan"      # 治疗方案制定
    MEDICATION = "medication"              # 药物选择与给药
    WORKFLOW = "workflow"                  # 治疗流程/步骤


# 根据任务类型获取对应的prompt
GENERATOR_PROMPTS = {
    TreatmentTaskType.TREATMENT_PLAN: TREATMENT_GENERATOR_PROMPT,
    TreatmentTaskType.MEDICATION: MEDICATION_GENERATOR_PROMPT,
    TreatmentTaskType.WORKFLOW: WORKFLOW_GENERATOR_PROMPT,
}

REFLECTOR_PROMPTS = {
    TreatmentTaskType.TREATMENT_PLAN: TREATMENT_REFLECTOR_PROMPT,
    TreatmentTaskType.MEDICATION: MEDICATION_REFLECTOR_PROMPT,
    TreatmentTaskType.WORKFLOW: TREATMENT_REFLECTOR_PROMPT,  # 复用通用模板
}


def get_prompts_for_task(task_type: str) -> tuple:
    """
    根据任务类型获取对应的 Generator 和 Reflector prompts。
    
    Parameters
    ----------
    task_type : str
        任务类型，来自 TreatmentTaskType
    
    Returns
    -------
    tuple of (generator_prompt, reflector_prompt, curator_prompt)
    """
    gen_prompt = GENERATOR_PROMPTS.get(task_type, TREATMENT_GENERATOR_PROMPT)
    ref_prompt = REFLECTOR_PROMPTS.get(task_type, TREATMENT_REFLECTOR_PROMPT)
    cur_prompt = TREATMENT_CURATOR_PROMPT
    
    return gen_prompt, ref_prompt, cur_prompt

