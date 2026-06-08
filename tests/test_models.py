"""Tests for :mod:`crier.models` — the ``ModelSpec`` and preset registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from crier import ModelSpec, list_presets, resolve_preset


def test_modelspec_requires_repo_id_or_local_path() -> None:
    with pytest.raises(ValueError, match="repo_id or local_path"):
        ModelSpec(name="x", backend="cpu")


def test_modelspec_rejects_both_repo_id_and_local_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot set both"):
        ModelSpec(
            name="x",
            backend="cpu",
            repo_id="hub/repo",
            local_path=tmp_path,
        )


def test_modelspec_local_path_only_is_valid(tmp_path: Path) -> None:
    spec = ModelSpec(name="x", backend="cpu", local_path=tmp_path)
    assert spec.local_path == tmp_path
    assert spec.repo_id is None


def test_list_presets_returns_sorted_pairs() -> None:
    presets = list_presets()
    assert all(isinstance(p, tuple) and len(p) == 2 for p in presets)
    assert presets == sorted(presets)


def test_resolve_preset_known_pair() -> None:
    spec = resolve_preset("phi-3.5-mini-instruct", "cpu")
    assert spec.name == "phi-3.5-mini-instruct"
    assert spec.backend == "cpu"
    assert spec.repo_id is not None


def test_resolve_preset_unknown_pair_lists_alternatives() -> None:
    with pytest.raises(KeyError, match="No preset"):
        resolve_preset("phi-3.5-mini-instruct", "this-backend-does-not-exist")


def test_resolve_preset_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        resolve_preset("no-such-model", "cpu")
