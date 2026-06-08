"""Backend registry — discovery, vendor-aware ordering, selection."""

from __future__ import annotations

import platform
import subprocess
from functools import lru_cache

from .backend import Backend
from .backends import (
    CoreMLBackend,
    CpuBackend,
    CudaBackend,
    DirectMLBackend,
    OpenVINOBackend,
    QnnBackend,
    RyzenAIBackend,
)
from .errors import BackendDependencyError, BackendUnavailableError, ConfigurationError


def list_backends() -> list[Backend]:
    """Return one fresh instance of every known backend.

    Order is intentionally undefined here; selection ordering is the
    registry's job, not the list's.
    """
    return [
        CpuBackend(),
        DirectMLBackend(),
        CoreMLBackend(),
        RyzenAIBackend(),
        OpenVINOBackend(),
        QnnBackend(),
        CudaBackend(),
    ]


def _by_name() -> dict[str, Backend]:
    return {b.capability.name: b for b in list_backends()}


def get_backend(name: str) -> Backend:
    """Return the backend with the given name, or raise ``ConfigurationError``."""
    backends = _by_name()
    if name not in backends:
        raise ConfigurationError(
            f"Unknown accelerator {name!r}. Known: {sorted(backends)}"
        )
    return backends[name]


# --- Host detection -------------------------------------------------------


@lru_cache(maxsize=1)
def detect_cpu_vendor() -> str:
    """Best-effort CPU vendor string. Empty when unknown."""
    system = platform.system()
    if system == "Darwin":
        # All modern Mac is Apple Silicon or Intel; cpuinfo is patchy.
        machine = platform.machine().lower()
        return "Apple" if machine in {"arm64", "aarch64"} else "GenuineIntel"
    if system == "Windows":
        # PROCESSOR_IDENTIFIER usually carries the vendor in the leading word.
        import os

        ident = (os.environ.get("PROCESSOR_IDENTIFIER") or "").lower()
        if "amd" in ident:
            return "AuthenticAMD"
        if "intel" in ident:
            return "GenuineIntel"
        if "qualcomm" in ident or "snapdragon" in ident or "arm" in ident:
            return "Qualcomm"
        return ""
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("vendor_id"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            return ""
        try:
            out = subprocess.run(
                ["uname", "-m"], capture_output=True, text=True, timeout=2
            ).stdout.strip()
            if out in {"aarch64", "arm64"}:
                return "Qualcomm"  # best guess on Linux ARM
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def _preferred_order(vendor: str, system: str) -> list[str]:
    """Backend preference order for an (os, vendor) pair.

    Order matters: we try them top-to-bottom and the first one whose probe
    succeeds wins.
    """
    if system == "Windows":
        if vendor == "AuthenticAMD":
            return ["ryzenai", "directml", "cpu"]
        if vendor == "GenuineIntel":
            return ["openvino", "directml", "cpu"]
        if vendor == "Qualcomm":
            return ["qnn", "directml", "cpu"]
        return ["directml", "cpu"]
    if system == "Darwin":
        return ["coreml", "cpu"]
    # Linux and anything else
    if vendor == "GenuineIntel":
        return ["openvino", "cpu"]
    if vendor == "AuthenticAMD":
        return ["ryzenai", "cpu"]
    return ["cpu"]


def available_backends() -> list[Backend]:
    """Return all backends whose package + runtime probe succeed."""
    return [b for b in list_backends() if b.probe().package_installed and b.probe().initialisable]


def select_backend(
    accelerator: str = "auto",
    *,
    require_acceleration: bool = False,
) -> tuple[Backend, list[tuple[str, str]], str | None]:
    """Pick a backend.

    Returns ``(backend, attempted, fallback_reason)``.

    ``attempted`` is the list of ``(backend_name, reject_reason)`` pairs we
    tried before settling. ``fallback_reason`` is non-None whenever the
    chosen backend is the CPU fallback after some non-CPU option failed.

    When ``accelerator`` is anything other than ``"auto"``, that backend
    must work or we raise. When ``"auto"`` and ``require_acceleration`` is
    True, we refuse to silently land on CPU and raise instead.
    """
    backends = _by_name()
    attempted: list[tuple[str, str]] = []

    if accelerator != "auto":
        backend = get_backend(accelerator)
        result = backend.probe()
        if not result.package_installed:
            raise BackendDependencyError(
                f"Backend {accelerator!r}: {result.detail} "
                f"Hint: {result.install_hint}"
            )
        if not result.initialisable:
            raise BackendUnavailableError(
                f"Backend {accelerator!r}: {result.detail}"
            )
        return backend, attempted, None

    system = platform.system()
    vendor = detect_cpu_vendor()
    order = _preferred_order(vendor, system)

    for name in order:
        if name not in backends:
            continue
        candidate = backends[name]
        result = candidate.probe()
        if result.package_installed and result.initialisable:
            fallback_reason = None
            if name == "cpu" and any(n != "cpu" for n in order):
                tried_non_cpu = [n for n, _ in attempted]
                if tried_non_cpu:
                    fallback_reason = (
                        "Fell back to CPU because: "
                        + "; ".join(f"{n}={r}" for n, r in attempted)
                    )
                    if require_acceleration:
                        raise BackendUnavailableError(
                            "require_acceleration=True but no accelerated "
                            f"backend is available. {fallback_reason}"
                        )
            return candidate, attempted, fallback_reason
        attempted.append((name, result.detail))

    raise BackendUnavailableError(
        f"No backend available on this host. Tried: {attempted}"
    )
