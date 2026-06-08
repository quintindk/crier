"""Capability probing — the diagnostic surface."""

from __future__ import annotations

from .registry import list_backends
from .types import ProbeResult


def probe() -> list[ProbeResult]:
    """Probe every known backend and return one ``ProbeResult`` per backend.

    Pure: no model loading, no network. Safe to call at any time. Intended
    to be the first thing a caller (or ``crier doctor``) runs when
    debugging a missing accelerator.
    """
    return [b.probe() for b in list_backends()]
