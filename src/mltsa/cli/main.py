"""Minimal CLI entry point for the mltsa package."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence

from mltsa import __version__


def build_parser() -> ArgumentParser:
    """Create the root argument parser for the project CLI."""
    parser = ArgumentParser(
        prog="mltsa",
        description=(
            "Machine Learning Transition State Analysis toolkit. "
            "Scientific subcommands will arrive in future milestones."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    parser.parse_args(list(argv) if argv is not None else None)
    parser.print_help()
    return 0
