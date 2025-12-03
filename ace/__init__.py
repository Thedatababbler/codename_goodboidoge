"""Agentic Context Engineering (ACE) reproduction framework."""

from .playbook import GuidelinePlaybook, DeltaOperation, DeltaBatch, Playbook
from .delta import DeltaOperation, DeltaBatch
from .llm import LLMClient, DummyLLMClient, TransformersLLMClient, OpenAIClient
from .roles import (
    Generator,
    Reflector,
    Curator,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
    Retriever
)
from .adaptation import (
    OfflineAdapter,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    AdapterStepResult,
)

__all__ = [
    # 基础组件
    "Playbook",
    "GuidelinePlaybook",
    "DeltaOperation",
    "DeltaBatch",
    "LLMClient",
    "DummyLLMClient",
    "TransformersLLMClient",
    "OpenAIClient",
    "Generator",
    "Reflector",
    "Curator",
    "GeneratorOutput",
    "ReflectorOutput",
    "CuratorOutput",
    "OfflineAdapter",
    "OnlineAdapter",
    "Sample",
    "TaskEnvironment",
    "EnvironmentResult",
    "AdapterStepResult",
    "Retriever",
]

# 医疗领域专用组件 - 延迟导入以避免循环依赖
def __getattr__(name):
    """Lazy import for medical components."""
    medical_exports = {
        "MedicalReflector",
        "MedicalCurator", 
        "MedicalReflectorOutput",
        "MedicalCuratorOutput",
        "BulletTag",
        "ProposedOperation",
        "apply_bullet_tags",
        "create_medical_agents",
        "MEDICAL_REFLECTOR_PROMPT",
        "MEDICAL_CURATOR_PROMPT",
    }
    if name in medical_exports:
        from . import medical_roles
        return getattr(medical_roles, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
