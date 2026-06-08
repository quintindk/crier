"""Thin adapter over ``onnxruntime_genai``.

All ORT GenAI usage funnels through this module. Tests stub a single
boundary (``RuntimeAdapter``) instead of monkey-patching every backend.

The runtime is constructed lazily so importing Crier does not pull in
``onnxruntime_genai``. Backends call ``load_runtime`` only when an ``LLM``
is actually being instantiated.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .errors import GenerationError, ModelIncompatibleError
from .types import Chunk, GenerationConfig, Message, Response, Usage

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class RuntimeAdapter(Protocol):
    """The minimum surface a backend exposes to the LLM facade.

    This is the seam between Crier's pure-Python facade and the native
    inference runtime. The default implementation wraps
    ``onnxruntime_genai``; tests provide a fake implementation that
    captures calls and returns scripted outputs.
    """

    backend_name: str
    execution_provider: str
    device: str
    model_path: str

    def generate(
        self, messages: list[Message], config: GenerationConfig
    ) -> Response: ...

    def stream(
        self, messages: list[Message], config: GenerationConfig
    ) -> Iterator[Chunk]: ...

    def close(self) -> None: ...


# --- Default ORT GenAI implementation -------------------------------------


@dataclass
class OrtGenAIAdapter:
    """Concrete adapter backed by ``onnxruntime_genai``.

    Constructed by ``Backend.create_runtime``. Backends provide the
    ``provider_options`` dict that pins the EP (e.g. DmlExecutionProvider,
    OpenVINOExecutionProvider, VitisAIExecutionProvider).
    """

    backend_name: str
    execution_provider: str
    device: str
    model_path: str
    provider_options: dict[str, Any] = field(default_factory=dict)

    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _tokenizer_stream: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import onnxruntime_genai as og
        except ImportError as exc:  # pragma: no cover - exercised when EP wheel missing
            raise ModelIncompatibleError(
                "onnxruntime_genai is not installed for backend "
                f"{self.backend_name!r}. Install the matching extra "
                f"(see README) and try again."
            ) from exc

        try:
            self._model = og.Model(self.model_path)
            self._tokenizer = og.Tokenizer(self._model)
            self._tokenizer_stream = self._tokenizer.create_stream()
        except Exception as exc:  # pragma: no cover - hardware-dependent
            raise ModelIncompatibleError(
                f"Failed to load model at {self.model_path!r} with backend "
                f"{self.backend_name!r}: {exc}"
            ) from exc

    # -- Public adapter API ---------------------------------------------

    def generate(self, messages: list[Message], config: GenerationConfig) -> Response:
        chunks = list(self.stream(messages, config))
        text = "".join(c.text for c in chunks)
        last = chunks[-1] if chunks else None
        return Response(
            text=text,
            usage=Usage(prompt_tokens=0, completion_tokens=len(chunks)),
            stop_reason=(last.stop_reason if last and last.stop_reason else "stop"),
        )

    def stream(
        self, messages: list[Message], config: GenerationConfig
    ) -> Iterator[Chunk]:
        try:
            import onnxruntime_genai as og
        except ImportError as exc:  # pragma: no cover
            raise GenerationError("onnxruntime_genai vanished mid-flight") from exc

        prompt = self._render_prompt(messages)
        input_tokens = self._tokenizer.encode(prompt)

        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=len(input_tokens) + config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
        )

        generator = og.Generator(self._model, params)
        generator.append_tokens(input_tokens)

        try:
            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                decoded = self._tokenizer_stream.decode(new_token)
                if decoded:
                    yield Chunk(text=decoded)
            yield Chunk(text="", is_final=True, stop_reason="stop")
        except Exception as exc:  # pragma: no cover - hardware-dependent
            raise GenerationError(str(exc)) from exc
        finally:
            del generator

    def close(self) -> None:
        # ORT GenAI cleans up via GC on the underlying handles; we drop refs
        # to make that deterministic.
        self._model = None
        self._tokenizer = None
        self._tokenizer_stream = None

    # -- Helpers --------------------------------------------------------

    def _render_prompt(self, messages: list[Message]) -> str:
        # ORT GenAI's Tokenizer does not currently expose Jinja chat
        # templating. Until it does, we apply a permissive fallback
        # template that most instruct-tuned ONNX bundles tolerate.
        parts = []
        for msg in messages:
            parts.append(f"<|{msg.role}|>\n{msg.content}<|end|>")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)


def load_runtime(
    *,
    backend_name: str,
    execution_provider: str,
    device: str,
    model_path: str | Path,
    provider_options: dict[str, Any] | None = None,
) -> RuntimeAdapter:
    """Construct the default ORT GenAI adapter.

    Backends call this from ``create_runtime``. Tests bypass it by passing a
    fake adapter directly to the ``LLM`` constructor via ``LLM._for_test``.
    """

    return OrtGenAIAdapter(
        backend_name=backend_name,
        execution_provider=execution_provider,
        device=device,
        model_path=str(model_path),
        provider_options=provider_options or {},
    )
