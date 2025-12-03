# Agentic Context Engineering (ACE) – Method Outline

This document captures the methodological details from *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models* (arXiv:2510.04618) required for reproduction.

## Core Concepts
- **Context-as-Playbook**: Instead of a monolithic prompt, ACE maintains a structured playbook composed of *bullets*. Each bullet owns:
  - `metadata`: unique identifier plus counters tracking how often it was marked helpful/harmful.
  - `content`: a focused strategy, common error, API schema reminder, verification checklist, etc.
- **Delta Updates**: Adaptation is carried out via small “delta” updates (sets of candidate bullets) merged into the playbook. Only localized edits occur—no full rewrites.
- **Grow-and-Refine**: The playbook grows via new bullets and accumulates signals in metadata. Periodically the curator performs refinement (deduplication with semantic comparison, counter adjustments, pruning).

## Agentic Roles
- **Generator**:
  - Receives the current playbook plus any reflective feedback.
  - Produces reasoning trajectories (e.g., code executions, step-by-step answers).
  - Tags which bullet IDs were referenced/helpful during generation.
- **Reflector**:
  - Observes generator trajectory, environment feedback, ground-truth when available.
  - Diagnoses errors, classifies bullet contributions (`helpful`, `harmful`, `neutral`).
  - Produces structured insight—root cause, corrective actions, key takeaways.
  - Can iterate up to 5 refinement rounds (paper setting).
- **Curator**:
  - Consumes reflections + current playbook stats.
  - Emits explicit operations in JSON:
    - `ADD`: append bullet to a section.
    - `UPDATE`: mutate metadata or content.
    - `TAG`: adjust helpful/harmful counters.
    - `REMOVE`: prune redundant or harmful bullets (implicit via operations).
  - Operates under token budget awareness, preserving interpretability and avoiding collapse.

All three roles share the same base LLM (DeepSeek-V3.1 in the paper) to attribute gains purely to context engineering.

## Adaptation Loops
- **Offline Adaptation**:
  1. Iterate over training samples for up to 5 epochs.
  2. For each sample: generator produces answer → reflector analyzes → curator emits delta.
  3. Merge delta into playbook; optionally perform deduplication.
- **Online Adaptation**:
  - Similar loop but executed on the test stream. After each prediction, update playbook before the next sample.
  - In ground-truth-free settings, rely on natural execution feedback (unit tests, environment signals).

## Feedback Signals
- AppWorld agents: relies on code execution traces, unit-test logs, API responses, task completions.
- Financial benchmarks (FiNER, Formula): uses ground-truth labels when available; otherwise derives signals from execution correctness.
- ACE performance degrades if no reliable feedback exists—curation quality hinges on the reflector’s diagnostic power.

## Implementation Requirements
- **Playbook Store**: persistent data structure containing sections, bullets, metadata counters, and embeddings for deduplication.
- **LLM Interface**: pluggable client shared by generator/reflector/curator, supporting different prompts and decoding parameters.
- **Trajectory Logger**: capture model reasoning, API/tool interactions, execution results, and label availability for reflector inputs.
- **Delta Merger**: deterministic logic to apply curator operations, update counters, enforce uniqueness, and trigger grow-and-refine policies.
- **Training Driver**: orchestrates offline/online loops, batching (paper uses batch size 1), and manages epoch limits.
- **Evaluation Harness**: compute benchmark metrics (TGC/SGC for AppWorld; accuracy for FiNER/Formula) using the base ReAct agent or domain-specific solver augmented with the ACE playbook.

## Hyperparameters from the Paper
- Base model: DeepSeek-V3.1 (non-thinking mode).
- Batch size: 1 delta per sample.
- Reflector refinement rounds: up to 5.
- Offline epochs: up to 5.
- Online mode: sequential processing of test samples.
- Grow-and-refine trigger: after each delta or lazily upon context overflow (implementation-dependent).

These notes will guide the implementation of a faithful ACE reproduction.
