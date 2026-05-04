from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from djinni_openai_ner import PROMPT_VERSION, refine_annotation, stringify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply deterministic cleanup rules to OpenAI weak-label annotations."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL annotations.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL annotations.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output} already exists. Pass --overwrite to replace it.")
        args.output.unlink()

    rows = 0
    entities_before = 0
    entities_after = 0
    docs_changed = 0
    dropped_by_label: Counter[str] = Counter()

    with args.input.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            original_entities = [
                entity
                for entity in (row.get("entities") or [])
                if isinstance(entity, dict)
            ]
            cleaned_annotation = refine_annotation({"entities": original_entities})
            cleaned_entities = cleaned_annotation["entities"]

            if len(cleaned_entities) != len(original_entities):
                docs_changed += 1
                kept_keys = {
                    (
                        stringify(entity.get("label")).upper(),
                        stringify(entity.get("text")),
                        int(entity.get("start", -1)),
                        int(entity.get("end", -1)),
                    )
                    for entity in cleaned_entities
                }
                for entity in original_entities:
                    key = (
                        stringify(entity.get("label")).upper(),
                        stringify(entity.get("text")),
                        int(entity.get("start", -1)),
                        int(entity.get("end", -1)),
                    )
                    if key not in kept_keys:
                        dropped_by_label[stringify(entity.get("label")).upper()] += 1

            row["entities"] = cleaned_entities
            row["cleanup_version"] = PROMPT_VERSION
            write_jsonl(args.output, row)

            rows += 1
            entities_before += len(original_entities)
            entities_after += len(cleaned_entities)

    logging.info(
        "Cleaned %s documents. Entities: %s -> %s. Changed docs: %s",
        rows,
        entities_before,
        entities_after,
        docs_changed,
    )
    if dropped_by_label:
        logging.info("Dropped entities by label: %s", dict(dropped_by_label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
