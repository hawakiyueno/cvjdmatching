from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from onet_mapping import (
    OnetMapper,
    load_onet_index,
    map_record_entities,
    stringify,
    summarize_record_mappings,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map extracted Stage 1 entities to O*NET occupations and descriptors."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL with Stage 1 entities.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL with O*NET mappings.")
    parser.add_argument(
        "--onet-index",
        type=Path,
        default=Path("artifacts/onet_index.jsonl"),
        help="JSONL index created by prepare_onet_index.py.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional aggregate summary JSON path. Defaults to '<output>.summary.json'.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Maximum O*NET candidates per entity.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Minimum combined match score required to keep a candidate.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional Hugging Face encoder name for semantic reranking.",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Optional device override for the embedding model, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--lexical-candidate-limit",
        type=int,
        default=256,
        help="How many lexical candidates to keep before semantic reranking.",
    )
    parser.add_argument(
        "--min-token-overlap",
        type=int,
        default=1,
        help="Minimum token overlap required for lexical candidate expansion.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional limit for smoke runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_aggregate_summary(summary: dict[str, Any], mappings: list[dict[str, Any]]) -> None:
    for mapping in mappings:
        summary["total_entities"] += 1
        label = stringify(mapping.get("entity_label")).upper()
        stats = summary["by_label"].setdefault(label, {"entities": 0, "supported": 0, "mapped": 0})
        stats["entities"] += 1
        if mapping.get("supported"):
            summary["supported_entities"] += 1
            stats["supported"] += 1
        if mapping.get("candidates"):
            summary["mapped_entities"] += 1
            stats["mapped"] += 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output} already exists. Pass --overwrite to replace it.")
        args.output.unlink()

    summary_output = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
    if summary_output.exists() and args.overwrite:
        summary_output.unlink()

    entries = load_onet_index(args.onet_index)
    logging.info("Loaded %s O*NET entries from %s.", len(entries), args.onet_index)

    mapper = OnetMapper(
        entries,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        lexical_candidate_limit=args.lexical_candidate_limit,
        min_token_overlap=args.min_token_overlap,
    )

    aggregate_summary: dict[str, Any] = {
        "input": str(args.input),
        "output": str(args.output),
        "onet_index": str(args.onet_index),
        "documents": 0,
        "total_entities": 0,
        "supported_entities": 0,
        "mapped_entities": 0,
        "mapped_rate": 0.0,
        "by_label": {},
        "parameters": {
            "top_k": args.top_k,
            "min_score": args.min_score,
            "embedding_model": args.embedding_model,
            "embedding_device": args.embedding_device,
            "lexical_candidate_limit": args.lexical_candidate_limit,
            "min_token_overlap": args.min_token_overlap,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    for row in read_jsonl(args.input):
        if args.max_documents is not None and processed >= args.max_documents:
            break
        mappings = map_record_entities(row, mapper, top_k=args.top_k, min_score=args.min_score)
        row["onet_mappings"] = mappings
        row["onet_mapping_summary"] = summarize_record_mappings(mappings)
        append_jsonl(args.output, row)

        processed += 1
        aggregate_summary["documents"] = processed
        update_aggregate_summary(aggregate_summary, mappings)
        if processed % 250 == 0:
            logging.info("Mapped %s documents.", processed)

    aggregate_summary["mapped_rate"] = aggregate_summary["mapped_entities"] / max(
        aggregate_summary["supported_entities"],
        1,
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "Mapped %s documents. Supported entities: %s. Mapped entities: %s (%.4f).",
        aggregate_summary["documents"],
        aggregate_summary["supported_entities"],
        aggregate_summary["mapped_entities"],
        aggregate_summary["mapped_rate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
