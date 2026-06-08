"""Ryzen AI (XDNA 2) backend.

Targets AMD Ryzen AI 300 series and newer with the hybrid NPU+iGPU
recipe. Requires the AMD Ryzen AI Software stack to be installed
system-wide (XDNA driver + OGA runtime); the pip extra only installs the
matching ``onnxruntime_genai`` build.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .._ort_adapter import RuntimeAdapter, load_runtime
from ..backend import Backend, BackendCapability


class RyzenAIBackend(Backend):
    capability = BackendCapability(
        name="ryzenai",
        execution_provider="VitisAIExecutionProvider",
        required_package="onnxruntime_genai",
        install_extra="ryzenai",
        supported_oses=frozenset({"Windows", "Linux"}),
        preferred_for_vendors=frozenset({"AuthenticAMD"}),
    )

    def _probe_runtime(self) -> tuple[bool, str]:
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError:
            return False, "onnxruntime is not installed."
        providers = set(ort.get_available_providers())
        if "VitisAIExecutionProvider" not in providers:
            return (
                False,
                "VitisAIExecutionProvider not exposed by onnxruntime. "
                "Install Ryzen AI SW + XDNA driver.",
            )
        return True, "VitisAI EP registered."

    def create_runtime(self, model_path: Path) -> RuntimeAdapter:
        return load_runtime(
            backend_name=self.capability.name,
            execution_provider=self.capability.execution_provider,
            device="npu",
            model_path=model_path,
            provider_options={"hybrid": "true"},
        )
