"""The ``LLM`` public facade.

This is the only class most callers touch. Keep it small.

Concurrency contract: **one ``LLM`` supports one active generation at a
time.** Streaming and ``generate`` are not safe to interleave on the same
instance from multiple threads or tasks. Create more instances or wrap
calls in a lock if you need parallelism.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

from ._ort_adapter import RuntimeAdapter
from .backend import Backend
from .errors import ConfigurationError, ModelNotFoundError
from .models import ModelSpec, resolve_preset
from .registry import select_backend
from .types import BackendInfo, Chunk, GenerationConfig, Message, Response


def _default_cache_dir() -> Path:
    env = os.environ.get("CRIER_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "crier"


def _download_model(spec: ModelSpec) -> Path:
    """Resolve a ``ModelSpec`` to a local directory, downloading if needed."""
    if spec.local_path is not None:
        if not spec.local_path.exists():
            raise ModelNotFoundError(
                f"ModelSpec {spec.name!r} points at non-existent local_path "
                f"{spec.local_path}"
            )
        return spec.local_path

    if spec.repo_id is None:  # pragma: no cover - guarded by ModelSpec.__post_init__
        raise ConfigurationError(f"ModelSpec {spec.name!r} has neither repo_id nor local_path")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            "huggingface_hub is required to download model presets"
        ) from exc

    cache_dir = _default_cache_dir() / "models" / spec.name / spec.backend
    cache_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns: list[str] | None = None
    if spec.subfolder is not None:
        allow_patterns = [f"{spec.subfolder}/*"]
    if spec.allow_patterns is not None:
        allow_patterns = list(spec.allow_patterns)

    snapshot_path = snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
    )
    if spec.subfolder is not None:
        return Path(snapshot_path) / spec.subfolder
    return Path(snapshot_path)


@dataclass
class LLM:
    """A loaded model bound to a single backend.

    Construct via :meth:`LLM.load`, not directly.
    """

    info: BackendInfo
    _adapter: RuntimeAdapter
    _lock: threading.Lock

    # -- Construction ---------------------------------------------------

    @classmethod
    def load(
        cls,
        *,
        model: str | ModelSpec,
        accelerator: str = "auto",
        require_acceleration: bool = False,
    ) -> LLM:
        """Load a model.

        Parameters
        ----------
        model:
            Either a logical preset name (resolved against the active
            backend), an explicit :class:`ModelSpec`, or a filesystem path
            (``str`` or ``Path``).
        accelerator:
            ``"auto"`` (default) or one of: ``cpu``, ``directml``,
            ``coreml``, ``ryzenai``, ``openvino``, ``qnn``, ``cuda``.
        require_acceleration:
            If True and ``accelerator='auto'``, refuse to fall back to CPU
            once an accelerated backend has been tried and failed.
        """
        backend, attempted, fallback_reason = select_backend(
            accelerator=accelerator,
            require_acceleration=require_acceleration,
        )
        spec = _resolve_model(model, backend)
        model_path = _download_model(spec)
        adapter = backend.create_runtime(model_path)
        info = BackendInfo(
            name=backend.capability.name,
            execution_provider=backend.capability.execution_provider,
            device=adapter.device,
            model_name=spec.name,
            model_path=str(model_path),
            accelerated=backend.capability.name != "cpu",
            attempted=tuple(attempted),
            fallback_reason=fallback_reason,
        )
        return cls(info=info, _adapter=adapter, _lock=threading.Lock())

    @classmethod
    def _for_test(cls, *, info: BackendInfo, adapter: RuntimeAdapter) -> LLM:
        """Test-only constructor that bypasses backend selection + download."""
        return cls(info=info, _adapter=adapter, _lock=threading.Lock())

    # -- Generation -----------------------------------------------------

    def generate(self, messages: list[Message], config: GenerationConfig | None = None) -> Response:
        """Run a synchronous generation."""
        cfg = config or GenerationConfig()
        with self._lock:
            return self._adapter.generate(messages, cfg)

    def stream(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> Iterator[Chunk]:
        """Stream chunks synchronously."""
        cfg = config or GenerationConfig()
        with self._lock:
            yield from self._adapter.stream(messages, cfg)

    async def astream(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> AsyncIterator[Chunk]:
        """Async iterator over chunks.

        Implemented by hopping each chunk off the generation thread via
        ``asyncio.to_thread``. Good enough for v1 — replace with a proper
        thread pool + queue when we hit backpressure issues.
        """
        cfg = config or GenerationConfig()

        loop = asyncio.get_event_loop()
        # Run the sync generator in a worker thread, marshalling each chunk
        # back into the loop via run_coroutine_threadsafe → queue.
        queue: asyncio.Queue[Chunk | None] = asyncio.Queue()
        sentinel: Chunk | None = None

        def worker() -> None:
            try:
                with self._lock:
                    for chunk in self._adapter.stream(messages, cfg):
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

        thread = threading.Thread(target=worker, name="crier-astream", daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    return
                yield item
        finally:
            thread.join(timeout=5.0)

    async def agenerate(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> Response:
        """Async ``generate`` — runs the sync path in a worker thread."""
        cfg = config or GenerationConfig()
        return await asyncio.to_thread(self.generate, messages, cfg)

    def close(self) -> None:
        """Release the underlying model handles."""
        self._adapter.close()

    def __enter__(self) -> LLM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- helpers --------------------------------------------------------------


def _resolve_model(model: str | ModelSpec | Path, backend: Backend) -> ModelSpec:
    """Map the polymorphic ``model`` argument to a ``ModelSpec``."""
    if isinstance(model, ModelSpec):
        if model.backend != backend.capability.name:
            raise ConfigurationError(
                f"ModelSpec targets backend {model.backend!r} but Crier "
                f"selected {backend.capability.name!r}. Either pin "
                f"accelerator={model.backend!r} or supply a matching ModelSpec."
            )
        return model

    if isinstance(model, Path):
        return ModelSpec(
            name=model.name,
            backend=backend.capability.name,
            local_path=model,
        )

    # str: try preset first; fall back to filesystem path if it exists.
    try:
        return resolve_preset(model, backend.capability.name)
    except KeyError:
        as_path = Path(model)
        if as_path.exists():
            return ModelSpec(
                name=as_path.name,
                backend=backend.capability.name,
                local_path=as_path,
            )
        raise ModelNotFoundError(
            f"No preset {model!r} for backend {backend.capability.name!r} "
            f"and {model!r} is not a local path."
        ) from None
