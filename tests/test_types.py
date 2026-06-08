"""Tests for the data containers in :mod:`crier.types`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from crier import (
    BackendInfo,
    Chunk,
    GenerationConfig,
    Message,
    ProbeResult,
    Response,
)
from crier.types import Usage


def test_message_is_frozen() -> None:
    msg = Message(role="user", content="hi")
    with pytest.raises(FrozenInstanceError):
        msg.content = "ho"  # type: ignore[misc]


def test_generation_config_defaults() -> None:
    cfg = GenerationConfig()
    assert cfg.max_tokens == 512
    assert 0.0 < cfg.temperature <= 1.0
    assert cfg.stop == ()


def test_usage_total_tokens() -> None:
    u = Usage(prompt_tokens=10, completion_tokens=5)
    assert u.total_tokens == 15


def test_response_carries_text_and_usage() -> None:
    r = Response(text="hi", usage=Usage(1, 2))
    assert r.text == "hi"
    assert r.usage.total_tokens == 3
    assert r.stop_reason == "stop"


def test_chunk_default_is_not_final() -> None:
    c = Chunk(text="x")
    assert not c.is_final
    assert c.stop_reason is None


def test_backend_info_attempted_default() -> None:
    info = BackendInfo(
        name="cpu",
        execution_provider="CPUExecutionProvider",
        device="cpu",
        model_name="m",
        model_path="/m",
        accelerated=False,
    )
    assert info.attempted == ()
    assert info.fallback_reason is None


def test_probe_result_str_human_friendly() -> None:
    r = ProbeResult(
        backend="cpu",
        package_installed=True,
        package_name="onnxruntime_genai",
        initialisable=True,
        detail="ok",
    )
    assert "cpu" in str(r)
    assert "ok" in str(r)


def test_probe_result_str_missing_package() -> None:
    r = ProbeResult(
        backend="ryzenai",
        package_installed=False,
        package_name="onnxruntime_genai",
        initialisable=False,
        detail="not installed",
        install_hint="pip install crier[ryzenai]",
    )
    assert "missing" in str(r)
