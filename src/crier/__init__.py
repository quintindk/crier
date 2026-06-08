"""
Crier — embedded LLM inference for the Chamberlain architecture.

Crier is the fifth pillar of Chamberlain. It is a cross-platform Python
**library** (not a service) that runs small language models on the NPU /
GPU / CPU of the host machine via ONNX Runtime GenAI, behind a single
clean interface. Other pillars import Crier directly; there is no HTTP
hop and no Docker.

Architectural contract lives in ``chamberlain/specification.md`` §7.

Public surface
--------------

    from crier import LLM, Message, GenerationConfig

    llm = LLM.load(model="phi-3.5-mini-instruct", accelerator="auto")
    print(llm.info)

    reply = llm.generate(
        [
            Message(role="system", content="You are concise."),
            Message(role="user", content="Why is the sky blue?"),
        ],
        config=GenerationConfig(max_tokens=128, temperature=0.7),
    )
    print(reply.text)

    for chunk in llm.stream(messages, config):
        print(chunk.text, end="", flush=True)

Diagnostics:

    from crier import probe
    for result in probe():
        print(result)
"""

from __future__ import annotations

from .diagnostics import probe
from .errors import (
    BackendDependencyError,
    BackendUnavailableError,
    ConfigurationError,
    CrierError,
    GenerationError,
    ModelIncompatibleError,
    ModelNotFoundError,
)
from .llm import LLM
from .models import ModelSpec, list_presets, resolve_preset
from .registry import available_backends, list_backends, select_backend
from .types import (
    BackendInfo,
    Chunk,
    GenerationConfig,
    Message,
    ProbeResult,
    Response,
)

__all__ = [
    "LLM",
    "BackendDependencyError",
    "BackendInfo",
    "BackendUnavailableError",
    "Chunk",
    "ConfigurationError",
    "CrierError",
    "GenerationConfig",
    "GenerationError",
    "Message",
    "ModelIncompatibleError",
    "ModelNotFoundError",
    "ModelSpec",
    "ProbeResult",
    "Response",
    "available_backends",
    "list_backends",
    "list_presets",
    "probe",
    "resolve_preset",
    "select_backend",
]

__version__ = "0.1.0"
