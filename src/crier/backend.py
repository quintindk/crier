"""Backend abstract base class.

A backend is a thin object that knows:

1. Which Python package it needs (``required_package``) and how to install
   it (``install_extra``).
2. How to probe the host for usability (``probe``).
3. How to create a runtime adapter for a given model path
   (``create_runtime``).

Backends do not know anything about chat templates or generation loops —
that all lives in the runtime adapter so tests can mock the seam.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ._ort_adapter import RuntimeAdapter
from .types import ProbeResult


@dataclass(frozen=True)
class BackendCapability:
    """Static description of what a backend claims to support."""

    name: str
    execution_provider: str
    required_package: str
    install_extra: str
    supported_oses: frozenset[str]
    preferred_for_vendors: frozenset[str] = frozenset()


class Backend(ABC):
    """Abstract base for all execution-provider backends."""

    capability: BackendCapability

    # -- Probing --------------------------------------------------------

    def probe(self) -> ProbeResult:
        """Quick capability check used by ``crier.probe()`` and ``select_backend``.

        Default implementation:
          * verify Python package import
          * delegate to ``_probe_runtime`` for device/driver checks
        Subclasses override ``_probe_runtime`` for vendor-specific checks.
        """
        cap = self.capability
        try:
            importlib.import_module(cap.required_package)
        except ImportError:
            return ProbeResult(
                backend=cap.name,
                package_installed=False,
                package_name=cap.required_package,
                initialisable=False,
                detail=f"Python package {cap.required_package!r} not installed.",
                install_hint=f"pip install crier[{cap.install_extra}]",
            )

        ok, detail = self._probe_runtime()
        return ProbeResult(
            backend=cap.name,
            package_installed=True,
            package_name=cap.required_package,
            initialisable=ok,
            detail=detail,
            install_hint=None if ok else f"See README section on {cap.name!r}.",
        )

    def _probe_runtime(self) -> tuple[bool, str]:  # pragma: no cover - per-backend override
        """Vendor-specific runtime probe. Default: assume installed == usable."""
        return True, "Package installed."

    # -- Runtime --------------------------------------------------------

    @abstractmethod
    def create_runtime(self, model_path: Path) -> RuntimeAdapter:
        """Instantiate the runtime adapter for a given model path."""
