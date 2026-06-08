"""Tests for the exception taxonomy."""

from __future__ import annotations

import pytest

from crier import (
    BackendDependencyError,
    BackendUnavailableError,
    ConfigurationError,
    CrierError,
    GenerationError,
    ModelIncompatibleError,
    ModelNotFoundError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [
        BackendDependencyError,
        BackendUnavailableError,
        ConfigurationError,
        GenerationError,
        ModelIncompatibleError,
        ModelNotFoundError,
    ],
)
def test_every_specific_error_inherits_crier_error(exc_cls: type[Exception]) -> None:
    assert issubclass(exc_cls, CrierError)


def test_crier_error_is_distinct_from_builtin() -> None:
    assert not issubclass(CrierError, ValueError)
