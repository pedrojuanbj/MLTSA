"""Smoke tests for the package skeleton."""

from __future__ import annotations

import mltsa


def test_imports_package() -> None:
    """The new public package should be importable."""
    assert mltsa.__version__ == "0.1.0"
