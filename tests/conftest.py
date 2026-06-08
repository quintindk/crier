"""Shared pytest fixtures and the fake runtime adapter.

The fake adapter is the only seam tests need: it captures inputs, returns
scripted outputs, and lets us exercise every line of the LLM facade and
the registry without ever loading ONNX Runtime GenAI.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

from crier._ort_adapter import RuntimeAdapter
from crier.types import Chunk, GenerationConfig, Message, Response, Usage


@dataclass
class FakeAdapter:
    """In-memory stand-in for ``OrtGenAIAdapter``.

    Implements the ``RuntimeAdapter`` protocol. Records every call and
    yields the chunks set on it via :meth:`script`.
    """

    backend_name: str = "fake"
    execution_provider: str = "FakeExecutionProvider"
    device: str = "fake"
    model_path: str = "/tmp/fake-model"

    scripted_chunks: list[Chunk] = field(default_factory=list)
    calls: list[tuple[str, list[Message], GenerationConfig]] = field(default_factory=list)
    closed: bool = False

    def script(self, *texts: str, stop_reason: str = "stop") -> FakeAdapter:
        self.scripted_chunks = [Chunk(text=t) for t in texts]
        self.scripted_chunks.append(Chunk(text="", is_final=True, stop_reason=stop_reason))
        return self

    def generate(self, messages: list[Message], config: GenerationConfig) -> Response:
        self.calls.append(("generate", messages, config))
        body = "".join(c.text for c in self.scripted_chunks if not c.is_final)
        return Response(
            text=body,
            usage=Usage(prompt_tokens=len(messages), completion_tokens=len(self.scripted_chunks) - 1),
            stop_reason="stop",
        )

    def stream(self, messages: list[Message], config: GenerationConfig) -> Iterator[Chunk]:
        self.calls.append(("stream", messages, config))
        yield from self.scripted_chunks

    def close(self) -> None:
        self.closed = True


# Statically declare it satisfies the protocol — useful for mypy in IDE.
_check: type[RuntimeAdapter] = FakeAdapter  # type: ignore[assignment]


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    return FakeAdapter().script("Hello, ", "world!", "\n")
