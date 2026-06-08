"""Exception taxonomy for Crier.

A small, explicit set of errors so callers (notably Catchpole) can pattern-match
on category rather than parsing strings.
"""

from __future__ import annotations


class CrierError(Exception):
    """Base for all Crier-raised exceptions."""


class ConfigurationError(CrierError):
    """The caller supplied invalid configuration (bad accelerator name, etc.)."""


class BackendDependencyError(CrierError):
    """A backend's Python package isn't installed.

    Raised when, for example, ``onnxruntime-genai-directml`` is not on the
    Python path but the caller asked for the DirectML backend. The message
    always includes the exact pip command to install the missing extra.
    """


class BackendUnavailableError(CrierError):
    """A backend's package is installed but it cannot initialise on this host.

    Typical causes: missing system driver (XDNA, Intel NPU, QNN runtime),
    incompatible OS, or no compatible device present.
    """


class ModelNotFoundError(CrierError):
    """The requested model could not be located locally or downloaded."""


class ModelIncompatibleError(CrierError):
    """The model exists but cannot be loaded by the selected backend.

    Typical cause: the artifact targets a different execution provider than
    the one Crier picked (e.g. a Ryzen AI hybrid bundle handed to the CPU
    backend).
    """


class GenerationError(CrierError):
    """Generation failed mid-flight."""
