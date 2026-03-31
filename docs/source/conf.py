"""Configuration for the lightweight mltsa documentation site."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mltsa import __version__

project = "mltsa"
copyright = "2026, Pedro Buigues, Edina Rosta"
author = "Pedro Buigues, Edina Rosta"
release = __version__

extensions: list[str] = []
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_templates",
    "_autosummary",
    "demos",
    "examples.rst",
    "onedfunctions.rst",
    "sklearnfunctions.rst",
    "summary.rst",
    "tffunctions.rst",
    "trajanalysis.rst",
    "twodpots.rst",
    "twodpots_source.rst",
]

source_suffix = ".rst"
html_theme = "alabaster"
html_title = "mltsa"
