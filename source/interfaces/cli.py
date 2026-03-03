from __future__ import annotations

import argparse

from source.interfaces.cli_migrate import main as migrate_main
from source.interfaces.cli_prepare import build_parser as build_prepare_parser
from source.interfaces.cli_prepare import run_prepare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BookRecs command router")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_sub = sub.add_parser("prepare-data", help="Run dataset preparation pipeline")
    prepare_inner = build_prepare_parser()
    for action in prepare_inner._actions:
        if action.dest == "help":
            continue
        prepare_sub._add_action(action)

    sub.add_parser("migrate", help="Apply PostgreSQL migration file")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare-data":
        run_prepare(args)
        return
    if args.command == "migrate":
        migrate_main()
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
