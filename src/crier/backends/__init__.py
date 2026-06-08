"""Backend implementations.

Each backend is a tiny class that pins one ONNX Runtime execution provider.
The interesting logic lives in the registry (selection) and the adapter
(runtime). Keep these files boring.
"""

from __future__ import annotations

from .coreml import CoreMLBackend
from .cpu import CpuBackend
from .cuda import CudaBackend
from .directml import DirectMLBackend
from .openvino import OpenVINOBackend
from .qnn import QnnBackend
from .ryzenai import RyzenAIBackend

__all__ = [
    "CoreMLBackend",
    "CpuBackend",
    "CudaBackend",
    "DirectMLBackend",
    "OpenVINOBackend",
    "QnnBackend",
    "RyzenAIBackend",
]
