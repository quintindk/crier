"""Tests for backend selection / registry logic.

We patch ``Backend.probe`` per-test to simulate different host
capabilities, then assert the registry picks the right backend in the
right order. No ORT GenAI needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from crier import (
    BackendDependencyError,
    BackendUnavailableError,
    ConfigurationError,
    available_backends,
    list_backends,
    select_backend,
)
from crier.backend import Backend
from crier.registry import _preferred_order, detect_cpu_vendor, get_backend
from crier.types import ProbeResult


def _ok(backend_name: str) -> ProbeResult:
    return ProbeResult(
        backend=backend_name,
        package_installed=True,
        package_name="onnxruntime_genai",
        initialisable=True,
        detail="ok",
    )


def _missing(backend_name: str) -> ProbeResult:
    return ProbeResult(
        backend=backend_name,
        package_installed=False,
        package_name="onnxruntime_genai",
        initialisable=False,
        detail="package missing",
        install_hint="pip install crier[x]",
    )


def _unavailable(backend_name: str, why: str = "no driver") -> ProbeResult:
    return ProbeResult(
        backend=backend_name,
        package_installed=True,
        package_name="onnxruntime_genai",
        initialisable=False,
        detail=why,
    )


def test_list_backends_returns_one_per_name() -> None:
    names = {b.capability.name for b in list_backends()}
    assert names == {"cpu", "directml", "coreml", "ryzenai", "openvino", "qnn", "cuda"}


def test_get_backend_unknown_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="Unknown accelerator"):
        get_backend("imaginary")


def test_preferred_order_windows_amd_prefers_ryzenai() -> None:
    order = _preferred_order("AuthenticAMD", "Windows")
    assert order[0] == "ryzenai"
    assert order[-1] == "cpu"


def test_preferred_order_windows_intel_prefers_openvino() -> None:
    assert _preferred_order("GenuineIntel", "Windows")[0] == "openvino"


def test_preferred_order_windows_qualcomm_prefers_qnn() -> None:
    assert _preferred_order("Qualcomm", "Windows")[0] == "qnn"


def test_preferred_order_darwin_prefers_coreml() -> None:
    assert _preferred_order("Apple", "Darwin") == ["coreml", "cpu"]


def test_preferred_order_linux_unknown_vendor_is_cpu_only() -> None:
    assert _preferred_order("", "Linux") == ["cpu"]


def test_detect_cpu_vendor_is_a_string() -> None:
    # Just exercise the function: result depends on host.
    assert isinstance(detect_cpu_vendor(), str)


def test_select_backend_explicit_uninstalled_raises_dependency() -> None:
    with (
        patch.object(Backend, "probe", lambda self: _missing(self.capability.name)),
        pytest.raises(BackendDependencyError, match="ryzenai"),
    ):
        select_backend(accelerator="ryzenai")


def test_select_backend_explicit_unavailable_raises_unavailable() -> None:
    with (
        patch.object(Backend, "probe", lambda self: _unavailable(self.capability.name)),
        pytest.raises(BackendUnavailableError),
    ):
        select_backend(accelerator="ryzenai")


def test_select_backend_unknown_name_raises_configuration() -> None:
    with pytest.raises(ConfigurationError):
        select_backend(accelerator="not-a-backend")


def test_select_backend_auto_picks_first_available() -> None:
    # Force the host to be Windows AMD with everything available.
    with (
        patch("crier.registry.platform.system", return_value="Windows"),
        patch("crier.registry.detect_cpu_vendor", return_value="AuthenticAMD"),
        patch.object(Backend, "probe", lambda self: _ok(self.capability.name)),
    ):
        backend, attempted, fallback = select_backend()
        assert backend.capability.name == "ryzenai"
        assert attempted == []
        assert fallback is None


def test_select_backend_auto_falls_back_with_reason() -> None:
    def probe(self: Backend) -> ProbeResult:
        if self.capability.name in {"ryzenai", "directml"}:
            return _unavailable(self.capability.name, "no driver")
        return _ok(self.capability.name)

    with (
        patch("crier.registry.platform.system", return_value="Windows"),
        patch("crier.registry.detect_cpu_vendor", return_value="AuthenticAMD"),
        patch.object(Backend, "probe", probe),
    ):
        backend, attempted, fallback = select_backend()
        assert backend.capability.name == "cpu"
        assert [n for n, _ in attempted] == ["ryzenai", "directml"]
        assert fallback is not None and "ryzenai" in fallback


def test_select_backend_require_acceleration_refuses_cpu_fallback() -> None:
    def probe(self: Backend) -> ProbeResult:
        if self.capability.name == "cpu":
            return _ok("cpu")
        return _unavailable(self.capability.name, "no driver")

    with (
        patch("crier.registry.platform.system", return_value="Windows"),
        patch("crier.registry.detect_cpu_vendor", return_value="AuthenticAMD"),
        patch.object(Backend, "probe", probe),
        pytest.raises(BackendUnavailableError, match="require_acceleration"),
    ):
        select_backend(require_acceleration=True)


def test_available_backends_returns_only_probed_ok() -> None:
    def probe(self: Backend) -> ProbeResult:
        return _ok(self.capability.name) if self.capability.name == "cpu" else _missing(self.capability.name)

    with patch.object(Backend, "probe", probe):
        names = {b.capability.name for b in available_backends()}
        assert names == {"cpu"}
