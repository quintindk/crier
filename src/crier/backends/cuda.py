"""CUDA backend — discrete NVIDIA GPU.

Not an NPU, but included for symmetry. Useful for dev / CI machines.
"""

from __future__ import annotations

from pathlib import Path

from .._ort_adapter import RuntimeAdapter, load_runtime
from ..backend import Backend, BackendCapability


class CudaBackend(Backend):
    capability = BackendCapability(
        name="cuda",
        execution_provider="CUDAExecutionProvider",
        required_package="onnxruntime_genai_cuda",
        install_extra="cuda",
        supported_oses=frozenset({"Windows", "Linux"}),
    )

    def create_runtime(self, model_path: Path) -> RuntimeAdapter:
        return load_runtime(
            backend_name=self.capability.name,
            execution_provider=self.capability.execution_provider,
            device="cuda",
            model_path=model_path,
        )
