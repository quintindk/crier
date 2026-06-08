"""Tests for CPU model -> XDNA generation classification."""

from __future__ import annotations

import pytest

from crier._cpu_info import detect_xdna_generation


@pytest.mark.parametrize(
    "model_name",
    [
        "AMD Ryzen 7 PRO 7840HS w/ Radeon 780M Graphics",
        "AMD Ryzen 7 7840U",
        "AMD Ryzen 5 7640HS",
        "AMD Ryzen 9 7940HX",
        "AMD Ryzen 7 PRO 8845HS",
        "AMD Ryzen 9 8945HX",
        "AMD Ryzen 5 8645H",
    ],
)
def test_detects_phoenix_hawk_point_as_xdna1(model_name: str) -> None:
    assert detect_xdna_generation(model_name) == "xdna1"


@pytest.mark.parametrize(
    "model_name",
    [
        "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
        "AMD Ryzen AI 9 365",
        "AMD Ryzen AI 7 350",
        "AMD Ryzen AI 7 PRO 350",
        "AMD Ryzen AI MAX+ PRO 395",
    ],
)
def test_detects_ryzen_ai_brand_as_xdna2(model_name: str) -> None:
    assert detect_xdna_generation(model_name) == "xdna2"


@pytest.mark.parametrize(
    "model_name",
    [
        "Intel(R) Core(TM) i7-13700H",
        "Apple M3 Pro",
        "AMD Ryzen 9 5950X",
        "AMD EPYC 9554",
        "",
    ],
)
def test_unclassifiable_returns_none(model_name: str) -> None:
    assert detect_xdna_generation(model_name) is None


def test_none_input_when_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from crier import _cpu_info

    monkeypatch.setattr(_cpu_info, "_read_cpu_model_name", lambda: None)
    assert detect_xdna_generation() is None
