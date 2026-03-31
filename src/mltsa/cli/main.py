"""CLI entry point for the mltsa package."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence

from mltsa import __version__

from . import md as md_cli


def build_parser() -> ArgumentParser:
    """Create the root argument parser for the project CLI."""
    parser = ArgumentParser(
        prog="mltsa",
        description=(
            "Machine Learning Transition State Analysis toolkit. "
            "Use `mltsa md ...` or the dedicated `mltsa-md` command for MD workflows."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    arguments = list(argv) if argv is not None else None
    if arguments and arguments[0] == "md":
        return int(md_cli.main(arguments[1:]))

    parser = build_parser()
    parser.parse_args(arguments)
    parser.print_help()
    return 0
