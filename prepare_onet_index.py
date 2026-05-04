from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from onet_mapping import prepare_onet_index, write_onet_index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact JSONL index from official O*NET export files."
    )
    parser.add_argument(
        "--onet-dir",
        required=True,
        type=Path,
        help="Directory containing official O*NET text exports such as 'Occupation Data.txt'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/onet_index.jsonl"),
        help="Output JSONL index path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output index if it already exists.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output} already exists. Pass --overwrite to replace it.")
        args.output.unlink()
        summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
        if summary_path.exists():
            summary_path.unlink()

    entries, summary = prepare_onet_index(args.onet_dir)
    write_onet_index(args.output, entries, summary)
    logging.info(
        "Wrote %s O*NET entries to %s.",
        summary["entries"],
        args.output,
    )
    logging.info("Entry types: %s", json.dumps(summary["entry_types"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
