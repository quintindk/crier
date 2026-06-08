"""Tests for the ``LLM`` facade against a fake runtime adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from crier import (
    LLM,
    BackendInfo,
    ConfigurationError,
    GenerationConfig,
    Message,
    ModelNotFoundError,
    ModelSpec,
)
from crier.backends.cpu import CpuBackend
from crier.llm import _resolve_model


def _info(backend: str = "fake") -> BackendInfo:
    return BackendInfo(
        name=backend,
        execution_provider="FakeExecutionProvider",
        device="fake",
        model_name="phi-test",
        model_path="/tmp/phi-test",
        accelerated=False,
    )


# --- LLM facade ------------------------------------------------------------


def test_llm_generate_round_trips_text(fake_adapter) -> None:
    llm = LLM._for_test(info=_info(), adapter=fake_adapter)
    reply = llm.generate(
        [Message(role="user", content="hi")], GenerationConfig(max_tokens=10)
    )
    assert reply.text == "Hello, world!\n"
    assert reply.usage.completion_tokens == 3
    assert ("generate", [Message(role="user", content="hi")], GenerationConfig(max_tokens=10)) in fake_adapter.calls


def test_llm_stream_yields_chunks(fake_adapter) -> None:
    llm = LLM._for_test(info=_info(), adapter=fake_adapter)
    chunks = list(llm.stream([Message(role="user", content="hi")]))
    bodies = [c.text for c in chunks if not c.is_final]
    assert "".join(bodies) == "Hello, world!\n"
    assert chunks[-1].is_final
    assert chunks[-1].stop_reason == "stop"


def test_llm_default_generation_config_is_used(fake_adapter) -> None:
    llm = LLM._for_test(info=_info(), adapter=fake_adapter)
    llm.generate([Message(role="user", content="hi")])
    _, _, cfg = fake_adapter.calls[0]
    assert isinstance(cfg, GenerationConfig)


def test_llm_close_is_idempotent_and_propagates_to_adapter(fake_adapter) -> None:
    llm = LLM._for_test(info=_info(), adapter=fake_adapter)
    llm.close()
    llm.close()
    assert fake_adapter.closed is True


def test_llm_context_manager_closes_on_exit(fake_adapter) -> None:
    with LLM._for_test(info=_info(), adapter=fake_adapter) as llm:
        assert llm.info.name == "fake"
    assert fake_adapter.closed is True


# --- async API -------------------------------------------------------------


async def test_llm_agenerate_runs_in_thread(fake_adapter) -> None:
    llm = LLM._for_test(info=_info(), adapter=fake_adapter)
    reply = await llm.agenerate([Message(role="user", content="hi")])
    assert reply.text == "Hello, world!\n"


async def test_llm_astream_yields_chunks(fake_adapter) -> None:
    llm = LLM._for_test(info=_info(), adapter=fake_adapter)
    collected: list[str] = []
    async for chunk in llm.astream([Message(role="user", content="hi")]):
        if not chunk.is_final:
            collected.append(chunk.text)
    assert "".join(collected) == "Hello, world!\n"


# --- model resolution ------------------------------------------------------


def test_resolve_model_accepts_modelspec_matching_backend(tmp_path: Path) -> None:
    spec = ModelSpec(name="m", backend="cpu", local_path=tmp_path)
    resolved = _resolve_model(spec, CpuBackend())
    assert resolved is spec


def test_resolve_model_rejects_modelspec_for_wrong_backend(tmp_path: Path) -> None:
    spec = ModelSpec(name="m", backend="ryzenai", local_path=tmp_path)
    with pytest.raises(ConfigurationError, match="targets backend"):
        _resolve_model(spec, CpuBackend())


def test_resolve_model_accepts_local_path(tmp_path: Path) -> None:
    resolved = _resolve_model(tmp_path, CpuBackend())
    assert resolved.local_path == tmp_path
    assert resolved.backend == "cpu"


def test_resolve_model_unknown_string_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="not a local path"):
        _resolve_model("definitely-not-a-real-preset-name", CpuBackend())


def test_resolve_model_known_preset_returns_spec() -> None:
    resolved = _resolve_model("phi-3.5-mini-instruct", CpuBackend())
    assert resolved.name == "phi-3.5-mini-instruct"
    assert resolved.backend == "cpu"
