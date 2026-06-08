"""ModelSpec — the real contract for a downloadable / loadable model.

Logical names like ``phi-3.5-mini-instruct`` are *convenience presets* that
resolve to a per-backend ``ModelSpec``. Callers can also build a ``ModelSpec``
directly or pass a local path, both first-class supported paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# A model's chat template is a Python format string applied to a list of
# messages. We keep this very simple on purpose: ORT GenAI's tokenizer
# usually owns the *real* template. The template here is a fallback for
# bundles that lack one, and a way to override for testing.
#
# Each backend's ModelSpec can opt into using the tokenizer-bundled
# template (the default) or override it via ``chat_template``.


@dataclass(frozen=True)
class ModelSpec:
    """Pointer + metadata for a model artifact, scoped to a single backend.

    Attributes
    ----------
    name:
        Stable logical name, e.g. ``"phi-3.5-mini-instruct"``. Used for
        cache directory naming and diagnostics.
    backend:
        The Crier backend this spec targets (``"cpu"``, ``"directml"``,
        ``"ryzenai"`` etc.). A logical model resolves to a different
        ``ModelSpec`` per backend because the on-disk artifact differs.
    repo_id:
        Hugging Face repo id. ``None`` if the model is local-only.
    revision:
        Pinned git revision / tag. Always pin in presets for
        reproducibility.
    subfolder:
        Subfolder inside the repo holding the matching variant
        (HF repos often ship cpu/cuda/directml side-by-side).
    local_path:
        If set, Crier will use this directory directly and never download.
        Mutually exclusive with ``repo_id``.
    allow_patterns:
        Optional glob list passed to ``snapshot_download`` to avoid pulling
        the whole repo. ``None`` => download everything in ``subfolder``.
    chat_template:
        Optional override for the chat template. Most callers leave this
        unset and let the tokenizer's own template apply.
    quantization:
        Free-text label for diagnostics (e.g. ``"int4-awq"``).
    notes:
        Free-text caveat (e.g. licence, hybrid mode requirements).
    """

    name: str
    backend: str
    repo_id: str | None = None
    revision: str | None = None
    subfolder: str | None = None
    local_path: Path | None = None
    allow_patterns: tuple[str, ...] | None = None
    chat_template: str | None = None
    quantization: str | None = None
    notes: str | None = None
    extras: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.local_path is None and self.repo_id is None:
            raise ValueError(
                f"ModelSpec {self.name!r} must set either repo_id or local_path"
            )
        if self.local_path is not None and self.repo_id is not None:
            raise ValueError(
                f"ModelSpec {self.name!r} cannot set both repo_id and local_path"
            )


# --- Preset registry -------------------------------------------------------
#
# The preset table is intentionally small. It is the curated set of
# "we have tested this combo" presets, not a guarantee that every model
# is available on every backend. Missing combos raise ModelNotFoundError
# at resolution time with a clear "build your own ModelSpec" message.

_PRESETS: dict[tuple[str, str], ModelSpec] = {
    # Phi-3.5 mini — Microsoft official ONNX bundle.
    # Repo layout has two subfolders: cpu_and_mobile/ and gpu/. The "gpu"
    # bundle is consumed by both DirectML and CUDA EPs.
    ("phi-3.5-mini-instruct", "cpu"): ModelSpec(
        name="phi-3.5-mini-instruct",
        backend="cpu",
        repo_id="microsoft/Phi-3.5-mini-instruct-onnx",
        subfolder="cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4",
        quantization="int4-awq",
    ),
    ("phi-3.5-mini-instruct", "directml"): ModelSpec(
        name="phi-3.5-mini-instruct",
        backend="directml",
        repo_id="microsoft/Phi-3.5-mini-instruct-onnx",
        subfolder="gpu/gpu-int4-awq-block-128",
        quantization="int4-awq",
        notes="Microsoft 'gpu' bundle, consumed by DirectML EP.",
    ),
    ("phi-3.5-mini-instruct", "cuda"): ModelSpec(
        name="phi-3.5-mini-instruct",
        backend="cuda",
        repo_id="microsoft/Phi-3.5-mini-instruct-onnx",
        subfolder="gpu/gpu-int4-awq-block-128",
        quantization="int4-awq",
        notes="Microsoft 'gpu' bundle, consumed by CUDA EP.",
    ),
    # Phi-3.5 mini — AMD hybrid (NPU prefill + iGPU decode) for Ryzen AI 300.
    # The AMD HF repo is gated; users must `huggingface-cli login` and
    # accept the model card terms on huggingface.co first.
    ("phi-3.5-mini-instruct", "ryzenai"): ModelSpec(
        name="phi-3.5-mini-instruct",
        backend="ryzenai",
        repo_id="amd/Phi-3.5-mini-instruct-awq-asym-uint4-g128-lmhead-onnx-hybrid",
        quantization="int4-awq-hybrid",
        notes=(
            "Gated AMD repo — accept terms on HF and run `huggingface-cli login`. "
            "Requires Ryzen AI SW 1.3+ with XDNA driver. Hybrid NPU+iGPU."
        ),
    ),
}


def list_presets() -> list[tuple[str, str]]:
    """Enumerate ``(model_name, backend)`` pairs known to Crier."""
    return sorted(_PRESETS.keys())


def resolve_preset(model_name: str, backend: str) -> ModelSpec:
    """Return the curated ``ModelSpec`` for a logical model on a backend.

    Raises ``KeyError`` if no preset exists for the pair. Callers can catch
    and build their own ``ModelSpec`` instead.
    """
    spec = _PRESETS.get((model_name, backend))
    if spec is None:
        available = sorted({b for (n, b) in _PRESETS if n == model_name})
        raise KeyError(
            f"No preset for model {model_name!r} on backend {backend!r}. "
            f"Available backends for this model: {available or 'none'}. "
            "Build a ModelSpec yourself and pass it to LLM.load(model=...)."
        )
    return spec
