"""Tests for :mod:`crier.diagnostics` and the CLI's ``doctor`` subcommand."""

from __future__ import annotations

from unittest.mock import patch

from crier import probe
from crier.backend import Backend
from crier.cli import main
from crier.types import ProbeResult


def test_probe_returns_one_result_per_backend() -> None:
    results = probe()
    backend_names = {r.backend for r in results}
    assert {"cpu", "directml", "coreml", "ryzenai", "openvino", "qnn", "cuda"} <= backend_names


def test_probe_results_are_pure_no_exception_on_missing_packages() -> None:
    # On a clean CI host none of the EP packages are installed; this
    # should still complete without raising.
    for r in probe():
        assert isinstance(r, ProbeResult)


def test_cli_doctor_exits_zero_and_prints_table(capsys) -> None:
    rc = main(["doctor"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "backend" in captured.out
    assert "cpu" in captured.out


def test_cli_doctor_includes_install_hint_when_missing(capsys) -> None:
    def probe(self: Backend) -> ProbeResult:
        return ProbeResult(
            backend=self.capability.name,
            package_installed=False,
            package_name=self.capability.required_package,
            initialisable=False,
            detail="not installed",
            install_hint=f"pip install crier[{self.capability.install_extra}]",
        )

    with patch.object(Backend, "probe", probe):
        rc = main(["doctor"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "hint: pip install crier[cpu]" in captured.out
