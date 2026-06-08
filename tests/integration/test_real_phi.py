"""Integration tests — load a real model and generate.

Gated behind the ``CRIER_RUN_INTEGRATION`` env var so they never run on
the default CI matrix (they download a multi-GB model and require a
working EP). Run locally or via the ``integration`` GitHub Actions
workflow_dispatch only.

Usage::

    CRIER_RUN_INTEGRATION=1 pytest -m integration
"""

from __future__ import annotations

import os

import pytest

from crier import LLM, GenerationConfig, Message

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("CRIER_RUN_INTEGRATION") != "1",
        reason="Set CRIER_RUN_INTEGRATION=1 to run integration tests.",
    ),
]


def test_phi_cpu_generates_some_text() -> None:
    """The smallest reliable end-to-end check: load Phi on CPU, get tokens.

    This is intentionally tolerant of model quirks — we only assert that
    *something* came back and that backend metadata is honest.
    """
    with LLM.load(model="phi-3.5-mini-instruct", accelerator="cpu") as llm:
        assert llm.info.name == "cpu"
        assert llm.info.execution_provider == "CPUExecutionProvider"

        reply = llm.generate(
            [
                Message(role="system", content="You are concise."),
                Message(role="user", content="Say hello in five words."),
            ],
            GenerationConfig(max_tokens=64, temperature=0.0),
        )
        assert reply.text.strip() != ""
        assert reply.usage.completion_tokens > 0
