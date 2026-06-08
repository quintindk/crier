"""CPU backend — always available, the safe fallback."""

from __future__ import annotations

from pathlib import Path

from .._ort_adapter import RuntimeAdapter, load_runtime
from ..backend import Backend, BackendCapability


class CpuBackend(Backend):
    capability = BackendCapability(
        name="cpu",
        execution_provider="CPUExecutionProvider",
        required_package="onnxruntime_genai",
        install_extra="cpu",
        supported_oses=frozenset({"Windows", "Linux", "Darwin"}),
    )

    def create_runtime(self, model_path: Path) -> RuntimeAdapter:
        return load_runtime(
            backend_name=self.capability.name,
            execution_provider=self.capability.execution_provider,
            device="cpu",
            model_path=model_path,
        )
