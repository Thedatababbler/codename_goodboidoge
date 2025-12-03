"""
offline_medical_adaptation.py

An offline adaptation example built on ACE-open.
Demonstrates a medical QA environment with multi-epoch playbook refinement.
"""

import json
import os
from ace import (
    OpenAIClient, Playbook, DummyLLMClient, Generator, Reflector, Curator,
    OfflineAdapter, Sample, TaskEnvironment, EnvironmentResult,Retriever,
    GuidelinePlaybook, DeltaBatch, DeltaOperation
)
# from ace.llm_openai import OpenAIClient
# ---------------------------------------------------------------------
# Step 1. 定义一个医疗任务环境（Toy Medical QA）
# ---------------------------------------------------------------------
# class MedicalToyEnv(TaskEnvironment):
#     """
#     一个最简医疗 QA 环境。
#     模拟 "agent" 给出诊疗建议并由环境评估正误。
#     """
#     def evaluate(self, sample, generator_output):
#         gt = sample.ground_truth or ""
#         pred = generator_output.final_answer.strip().lower()

#         # 根据 ground truth 判断对错
#         correct = pred == gt.lower()
#         if correct:
#             feedback = f"✅ Correct. Adhered to guideline for {sample.question}."
#         else:
#             feedback = f"❌ Incorrect. Expected: {gt}, got: {pred}"

#         # EnvironmentResult 是 ACE 框架所需返回类型
#         return EnvironmentResult(
#             feedback=feedback,
#             ground_truth=gt
#         )

# ---------------------------------------------------------------------
# Step 2. 构造一个虚拟 LLM (Dummy) —— 仅作演示
# ---------------------------------------------------------------------
# client = DummyLLMClient()

# 三个 Agent 角色（Generator / Reflector / Curator）的模拟输出
# client.queue(json.dumps({
#     "reasoning": "Based on chest pain and cough, pneumonia is likely.",
#     "bullet_ids": [],
#     "final_answer": "pneumonia"
# }))
# client.queue(json.dumps({
#     "reasoning": "Compare prediction to guideline...",
#     "error_identification": "Missed differential of bronchitis.",
#     "root_cause_analysis": "Insufficient attention to fever pattern.",
#     "correct_approach": "Check temperature and sputum characteristics.",
#     "key_insight": "Always rule out bronchitis.",
#     "bullet_tags": ["diagnosis", "respiratory"]
# }))
# client.queue(json.dumps({
#     "reasoning": "Integrate insight into playbook.",
#     "operations": [{
#         "type": "ADD",
#         "section": "respiratory_guideline",
#         "content": "Rule out bronchitis before diagnosing pneumonia.",
#         "metadata": {"helpful": 1}
#     }]
# }))
# 一个面向医疗诊断任务的最小提示模板（仅含 playbook / context / query）
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
    "1) Return EXACTLY ONE valid JSON object and NOTHING ELSE (no prose, no markdown, no code fences).\n"
    "2) JSON keys (all required):\n"
    "   - \"reasoning\": a SHORT justification (1–3 sentences) that cites key evidence from the context (e.g., symptoms, signs, labs, imaging) that support the diagnosis.\n"
    "   - \"final_answer\": the single most likely diagnosis as a short phrase (e.g., \"acute appendicitis\"). No extra words, no differential list, no explanations.\n"
    "   - \"bullet_ids\": an array of strings referencing any useful playbook item IDs you used; if none, use an empty array [].\n"
    "3) Safety & scope: If information is clearly insufficient to make a safe diagnosis, set \"final_answer\" to \"insufficient information\" and keep \"bullet_ids\": [].\n"
    "4) Formatting: Use DOUBLE quotes for all JSON keys/strings. Do not include trailing commas. Do not add comments.\n"
    "\n"
    "OUTPUT JSON EXAMPLE (structure only; your content must reflect THIS task):\n"
    "{{\n"
    "  \"reasoning\": \"RLQ pain with positive Rovsing sign and ultrasound showing inflamed appendix support acute appendicitis.\",\n"
    "  \"final_answer\": \"acute appendicitis\",\n"
    "  \"bullet_ids\": [\"respiratory_guideline:note-42\"]\n"
    "}}\n"
    "\n"
    "Now produce ONLY the final JSON object for this case."
)

test_case = { "Objective_for_Doctor": "Evaluate and diagnose the patient presenting with a chronic lesion on the lower lip.",
 "Patient_Actor": { "Demographics": "58-year-old white male", 
 "History": "The patient reports a 3-month history of a painless lesion on his lower lip. He mentions a 20-year history of smoking one pack of cigarettes a day and has been working as a fruit picker for 25 years.", 
 "Symptoms": { "Primary_Symptom": "Painless lesion on the lower lip", "Secondary_Symptoms": [] }, "Past_Medical_History": "Hypertension, type 2 diabetes mellitus. Current medications include captopril and metformin.", "Social_History": "Smokes one pack of cigarettes a day, works outdoors as a fruit picker.", "Review_of_Systems": "Denies fever, weight loss, night sweats, or significant changes in appetite." }, 
 "Physical_Examination_Findings": { "Vital_Signs": { "Temperature": "36.8°C (98°F)", "Blood_Pressure": "130/85 mmHg", "Heart_Rate": "80 bpm", "Respiratory_Rate": "14 breaths/min" }, 
 "Oral_Examination": { "Oral_Cavity": "A single ulcer near the vermillion border of the lower lip. The ulcer is well-defined with a hard base and non-tender.", "Teeth_and_Gums": "No significant abnormalities noted.", "Other": "No lymphadenopathy." } }, 
 "Test_Results": { "Biopsy_of_Lesion": { "Findings": "Histopathology confirms squamous cell carcinoma." } }, 
 "Correct_Diagnosis": "Squamous cell carcinoma" } 


# llm = OpenAIClient(
#     model=os.getenv("OPENAI_MODEL", "gpt-4"),
#     temperature=0.0,
#     max_output_tokens=512,
#     max_retries=3,
# )

Playbook_library = '/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/OpenCE/data/guideline_dict.json'

# AgentRet = Retriever(Playbook_Library, 'dict', )
AgentRet = Retriever(
    json_path="data/guideline_dict.json",
    source="wikidoc",                       # 或 None 表示全部 source
    match_fields="title",          # 默认就是这个
    # embedding_fn=my_embedding_fn,
    # wikidoc_aliases={                       # 可选：自然语言 -> wikidoc 缩写
    #     "acute myocardial infarction": "AMI",
    #     "ami": "AMI",
    # },
)

guideline_text = AgentRet.retrieve_exact('Influenza').entries[0].text

pb = GuidelinePlaybook.from_markdown(
    guideline_id="influenza_overview_v1",
    title="Influenza Overview",
    text=guideline_text,
)

data = pb.to_hierarchical_dict()
# data["sections"][0]["bullets"][0]["id"] == 比如 "b-00001"

# 2）导出成 JSON 字符串（给大模型 / 前端 / 存盘）
json_str = pb.to_hierarchical_json()
print(json_str[:500])  # 看前 500 字符

import pdb;pdb.set_trace()
import re
from typing import Dict, Any, List, Optional
from ace.utils.utils import *



question= build_diagnosis_question(test_case)
# AgentGen = Generator()
AgentGen = Generator(llm, prompt_template=GENERATOR_PROMPT_MEDICAL)  # 你前面定义的医疗模板

# print(out.final_answer)
# AgentRef = Reflector()
# AgentCur = Curator()




# ---------------------------------------------------------------------
# Step 3. 初始化各角色组件
# ---------------------------------------------------------------------
# adapter = OfflineAdapter(
#     playbook=Playbook(),
#     generator=Generator(client),
#     reflector=Reflector(client),
#     curator=Curator(client),
# )

#step 1. retrieve playbook for a disease
# playbook = AgentRet.retrieve(query)
# playbook_obj = """
# You are reading a EHR document. Please first check symptom and give diagnosis based on the information provide.
# """



#step 2. Generator reasoning
# response = AgentGen.reason(playook, query)
response = AgentGen.generate(
    question=question,
    context="",
    playbook=playbook_obj,
)

print(response)
import pdb;pdb.set_trace()
# response {'pred': Answer, 
            # 'reasoning': trajectory,
#           'used bullet': bullet from playbook}
import system

# system.exit()
# #step 3. reflecting
# operation = AgentRef.reflect(response, playbook, ground_truth)
# # operation = {"type":(add, update, remove, tag), "section": which bullet or section to improve}

# #step 4. update playbook
# AgentCur(playbook, operation)

# Playbook_Library.update(playbook)


# # ---------------------------------------------------------------------
# # Step 4. 构建示例样本
# # ---------------------------------------------------------------------
# samples = [
#     Sample(question="Patient presents with chest pain and cough. What is the most likely diagnosis?",
#            ground_truth="pneumonia"),
#     Sample(question="Patient has sore throat and fever. What is the diagnosis?",
#            ground_truth="pharyngitis"),
# ]

# # ---------------------------------------------------------------------
# # Step 5. 运行离线强化循环
# # ---------------------------------------------------------------------
# print("=== [Start Offline Medical Adaptation] ===")
# adapter.run(samples, MedicalToyEnv(), epochs=2)
# print("=== [Finished] ===")

# # 保存最终 playbook
# adapter.playbook.save("medical_playbook.json")

# print("\n✅ Playbook saved to medical_playbook.json")
