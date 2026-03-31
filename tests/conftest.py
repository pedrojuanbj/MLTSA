"""Test configuration for local src-layout imports."""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture()
def workspace_tmp_dir() -> Path:
    """Provide a workspace-local temporary directory for file-based tests."""

    base_dir = Path(__file__).resolve().parents[1] / ".test_tmp"
    base_dir.mkdir(exist_ok=True)

    temp_dir = base_dir / f"mltsa-{uuid.uuid4().hex}"
    temp_dir.mkdir()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
