"""Core dataclasses used across Crier.

Kept dependency-free on purpose: they import nothing from ONNX Runtime so
callers can build / inspect requests and responses without pulling in the
heavy native stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    """A single chat turn.

    Crier accepts an explicit list of these and applies the model's prompt
    template at generation time. Callers should not pre-format prompt
    strings; let the ``ModelSpec`` template do it so swapping models stays
    a one-line change.
    """

    role: Role
    content: str


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters for a single generation call."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True)
class Usage:
    """Token accounting for a single generation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Response:
    """A completed (non-streaming) generation."""

    text: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: Literal["stop", "length", "error"] = "stop"


@dataclass(frozen=True)
class Chunk:
    """A streamed fragment."""

    text: str
    is_final: bool = False
    stop_reason: Literal["stop", "length", "error"] | None = None


@dataclass(frozen=True)
class BackendInfo:
    """Describes the backend an ``LLM`` is bound to.

    ``attempted`` lists every backend that was tried before the active one
    was chosen, with the reason each was rejected. This is the diagnostic
    surface that prevents silent CPU fallback from being invisible.
    """

    name: str
    execution_provider: str
    device: str
    model_name: str
    model_path: str
    accelerated: bool
    attempted: tuple[tuple[str, str], ...] = ()
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    """A single row of ``crier.probe()`` output."""

    backend: str
    package_installed: bool
    package_name: str
    initialisable: bool
    detail: str
    install_hint: str | None = None

    def __str__(self) -> str:  # human-friendly
        status = (
            "ok"
            if self.package_installed and self.initialisable
            else "missing-package"
            if not self.package_installed
            else "unavailable"
        )
        return f"{self.backend:>10}  {status:>16}  {self.detail}"


# A small structural type, kept loose because backends evolve.
ProviderOptions = dict[str, Any]
