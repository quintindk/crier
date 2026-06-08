"""OpenVINO backend — Intel CPU/GPU/NPU.

Requires the OpenVINO runtime in addition to ``onnxruntime_genai``;
both are pulled by ``pip install crier[openvino]`` but the NPU device
also requires the Intel NPU driver to be installed system-wide.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .._ort_adapter import RuntimeAdapter, load_runtime
from ..backend import Backend, BackendCapability


class OpenVINOBackend(Backend):
    capability = BackendCapability(
        name="openvino",
        execution_provider="OpenVINOExecutionProvider",
        required_package="onnxruntime_genai",
        install_extra="openvino",
        supported_oses=frozenset({"Windows", "Linux"}),
        preferred_for_vendors=frozenset({"GenuineIntel"}),
    )

    def _probe_runtime(self) -> tuple[bool, str]:
        try:
            ort = importlib.import_module("onnxruntime")
            importlib.import_module("openvino")
        except ImportError as exc:
            return False, f"Dependency missing: {exc}"
        providers = set(ort.get_available_providers())
        if "OpenVINOExecutionProvider" not in providers:
            return False, "OpenVINOExecutionProvider not registered."
        return True, "OpenVINO EP registered."

    def create_runtime(self, model_path: Path) -> RuntimeAdapter:
        return load_runtime(
            backend_name=self.capability.name,
            execution_provider=self.capability.execution_provider,
            device="NPU",
            model_path=model_path,
            provider_options={"device_type": "NPU"},
        )
