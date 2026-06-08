"""Qualcomm QNN backend — Snapdragon NPU."""

from __future__ import annotations

import importlib
from pathlib import Path

from .._ort_adapter import RuntimeAdapter, load_runtime
from ..backend import Backend, BackendCapability


class QnnBackend(Backend):
    capability = BackendCapability(
        name="qnn",
        execution_provider="QNNExecutionProvider",
        required_package="onnxruntime_genai",
        install_extra="qnn",
        supported_oses=frozenset({"Windows", "Linux"}),
        preferred_for_vendors=frozenset({"Qualcomm"}),
    )

    def _probe_runtime(self) -> tuple[bool, str]:
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError:
            return False, "onnxruntime not installed."
        providers = set(ort.get_available_providers())
        if "QNNExecutionProvider" not in providers:
            return False, "QNNExecutionProvider not registered."
        return True, "QNN EP registered."

    def create_runtime(self, model_path: Path) -> RuntimeAdapter:
        return load_runtime(
            backend_name=self.capability.name,
            execution_provider=self.capability.execution_provider,
            device="npu",
            model_path=model_path,
            provider_options={"backend_path": "QnnHtp.dll"},
        )
