"""CoreML backend — macOS, Apple Neural Engine on Apple Silicon.

Caveat: ORT GenAI's CoreML EP is workable but MLX is sharper on Apple
Silicon. We expose this for cross-platform symmetry; expect to revisit
with an MLX backend in a later release.
"""

from __future__ import annotations

from pathlib import Path

from .._ort_adapter import RuntimeAdapter, load_runtime
from ..backend import Backend, BackendCapability


class CoreMLBackend(Backend):
    capability = BackendCapability(
        name="coreml",
        execution_provider="CoreMLExecutionProvider",
        required_package="onnxruntime_genai",
        install_extra="coreml",
        supported_oses=frozenset({"Darwin"}),
        preferred_for_vendors=frozenset({"Apple"}),
    )

    def create_runtime(self, model_path: Path) -> RuntimeAdapter:
        return load_runtime(
            backend_name=self.capability.name,
            execution_provider=self.capability.execution_provider,
            device="ane",
            model_path=model_path,
        )
