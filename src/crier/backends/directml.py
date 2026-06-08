"""DirectML backend — Windows generic GPU/NPU passthrough."""

from __future__ import annotations

from pathlib import Path

from .._ort_adapter import RuntimeAdapter, load_runtime
from ..backend import Backend, BackendCapability


class DirectMLBackend(Backend):
    capability = BackendCapability(
        name="directml",
        execution_provider="DmlExecutionProvider",
        required_package="onnxruntime_genai",
        install_extra="directml",
        supported_oses=frozenset({"Windows"}),
        preferred_for_vendors=frozenset({"AuthenticAMD", "GenuineIntel"}),
    )

    def create_runtime(self, model_path: Path) -> RuntimeAdapter:
        return load_runtime(
            backend_name=self.capability.name,
            execution_provider=self.capability.execution_provider,
            device="dml",
            model_path=model_path,
        )
