from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset, load_dataset_builder
from prepare_seed_subset import clean_text as clean_seed_text
from prepare_seed_subset import is_it_row


CV_DATASET = "lang-uk/recruitment-dataset-candidate-profiles-english"
JD_DATASET = "lang-uk/recruitment-dataset-job-descriptions-english"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a reproducible 10k CV + 10k JD subset from Hugging Face using the datasets library."
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file.")
    parser.add_argument("--cv-limit", type=int, default=10_000, help="Number of CV rows to export.")
    parser.add_argument("--jd-limit", type=int, default=10_000, help="Number of JD rows to export.")
    parser.add_argument("--cv-offset", type=int, default=0, help="Start offset inside the CV dataset.")
    parser.add_argument("--jd-offset", type=int, default=0, help="Start offset inside the JD dataset.")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face datasets cache directory.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Optional Hugging Face token. Defaults to HF_TOKEN.")
    parser.add_argument("--it-only", action="store_true", help="Keep only records that look like IT roles. The script will scan further until it gathers enough accepted rows.")
    parser.add_argument("--it-strict", action="store_true", help="Use a stricter IT filter that removes ambiguous business/support roles unless the row has strong technical evidence.")
    parser.add_argument("--max-scan-cv", type=int, help="Optional maximum number of raw CV rows to scan after the offset.")
    parser.add_argument("--max-scan-jd", type=int, help="Optional maximum number of raw JD rows to scan after the offset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing partial output file.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
    if not verbose:
        for logger_name in ("httpx", "httpcore", "huggingface_hub", "fsspec"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"null", "nan", "none"}:
        return ""
    return text


def add_line(lines: list[str], label: str, value: Any) -> None:
    text = clean_text(value)
    if text:
        lines.append(f"{label}: {text}")


def compose_cv_text(row: dict[str, Any]) -> str:
    lines: list[str] = []
    add_line(lines, "Position", row.get("Position"))
    add_line(lines, "Primary Keyword", row.get("Primary Keyword"))
    add_line(lines, "English Level", row.get("English Level"))
    add_line(lines, "Experience Years", row.get("Experience Years"))

    cv_text = clean_text(row.get("CV"))
    moreinfo = clean_text(row.get("Moreinfo"))
    looking_for = clean_text(row.get("Looking For"))
    highlights = clean_text(row.get("Highlights"))

    if cv_text:
        lines.append(f"CV:\n{cv_text}")

    if len(cv_text) < 80:
        if moreinfo:
            lines.append(f"More Info:\n{moreinfo}")
        if looking_for:
            lines.append(f"Looking For:\n{looking_for}")
        if highlights:
            lines.append(f"Highlights:\n{highlights}")

    return "\n\n".join(lines).strip()


def compose_jd_text(row: dict[str, Any]) -> str:
    lines: list[str] = []
    add_line(lines, "Position", row.get("Position"))
    add_line(lines, "Company Name", row.get("Company Name"))
    add_line(lines, "Primary Keyword", row.get("Primary Keyword"))
    add_line(lines, "English Level", row.get("English Level"))
    add_line(lines, "Experience Years", row.get("Exp Years"))

    description = clean_text(row.get("Long Description"))
    if description:
        lines.append(f"Job Description:\n{description}")

    return "\n\n".join(lines).strip()


def normalize_cv_row(row: dict[str, Any], source_offset: int | None = None) -> dict[str, Any]:
    source_id = clean_text(row.get("id"))
    text = compose_cv_text(row)
    normalized = {
        "id": f"cv::{source_id}" if source_id else "",
        "source_id": source_id,
        "doc_type": "cv",
        "text": text,
        "source_dataset": CV_DATASET,
        "source_language": clean_text(row.get("CV_lang")) or "unknown",
        "position": clean_text(row.get("Position")),
        "primary_keyword": clean_text(row.get("Primary Keyword")),
        "english_level": clean_text(row.get("English Level")),
        "experience_years": clean_text(row.get("Experience Years")),
    }
    if source_offset is not None:
        normalized["source_offset"] = source_offset
    return normalized


def normalize_jd_row(row: dict[str, Any], source_offset: int | None = None) -> dict[str, Any]:
    source_id = clean_text(row.get("id"))
    text = compose_jd_text(row)
    normalized = {
        "id": f"jd::{source_id}" if source_id else "",
        "source_id": source_id,
        "doc_type": "jd",
        "text": text,
        "source_dataset": JD_DATASET,
        "source_language": clean_text(row.get("Long Description_lang")) or "unknown",
        "position": clean_text(row.get("Position")),
        "company_name": clean_text(row.get("Company Name")),
        "primary_keyword": clean_text(row.get("Primary Keyword")),
        "english_level": clean_text(row.get("English Level")),
        "experience_years": clean_text(row.get("Exp Years")),
        "published": clean_text(row.get("Published")),
    }
    if source_offset is not None:
        normalized["source_offset"] = source_offset
    return normalized


def parse_source_offset(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def inspect_existing_rows(path: Path) -> tuple[int, int, int | None, int | None]:
    cv_count = 0
    jd_count = 0
    cv_next_offset: int | None = None
    jd_next_offset: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            doc_type = row.get("doc_type")
            source_offset = parse_source_offset(row.get("source_offset"))
            if doc_type == "cv":
                cv_count += 1
                if source_offset is not None:
                    next_offset = source_offset + 1
                    cv_next_offset = max(cv_next_offset or 0, next_offset)
            elif doc_type == "jd":
                jd_count += 1
                if source_offset is not None:
                    next_offset = source_offset + 1
                    jd_next_offset = max(jd_next_offset or 0, next_offset)
    return cv_count, jd_count, cv_next_offset, jd_next_offset


def count_existing_rows(path: Path) -> tuple[int, int]:
    cv_count, jd_count, _, _ = inspect_existing_rows(path)
    return cv_count, jd_count


def write_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_total_rows(dataset_name: str, cache_dir: Path | None, token: str | None) -> int | None:
    builder = load_dataset_builder(dataset_name, cache_dir=str(cache_dir) if cache_dir else None, token=token)
    split_info = builder.info.splits.get("train")
    if split_info is None:
        return None
    return split_info.num_examples


def iter_dataset_rows(
    dataset_name: str,
    offset: int,
    limit: int | None,
    cache_dir: Path | None,
    token: str | None,
) -> Iterable[dict[str, Any]]:
    if limit is not None and limit <= 0:
        return []

    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir else None,
        token=token,
    )
    if offset > 0:
        dataset = dataset.skip(offset)
    if limit is None:
        return dataset
    return dataset.take(limit)


def looks_like_it(normalized_row: dict[str, Any], strict: bool = False) -> bool:
    cleaned_text = clean_seed_text(str(normalized_row.get("text") or ""))
    return is_it_row(normalized_row, cleaned_text, strict=strict)


def remaining_scan_budget(max_scan: int | None, base_offset: int, resume_offset: int) -> int | None:
    if max_scan is None:
        return None
    scanned_already = max(resume_offset - base_offset, 0)
    return max(max_scan - scanned_already, 0)


def iter_normalized_rows(
    dataset_name: str,
    offset: int,
    limit: int,
    cache_dir: Path | None,
    token: str | None,
    normalizer,
    it_only: bool = False,
    it_strict: bool = False,
    max_scan: int | None = None,
) -> Iterable[dict[str, Any]]:
    total_rows = get_total_rows(dataset_name, cache_dir=cache_dir, token=token)
    if total_rows is not None:
        logging.info("Dataset %s reports %s total rows", dataset_name, total_rows)

    if limit <= 0:
        return []

    raw_limit = max_scan
    if raw_limit is None and total_rows is not None:
        raw_limit = max(total_rows - offset, 0)

    yielded = 0
    scanned = 0
    for relative_index, row in enumerate(iter_dataset_rows(
        dataset_name=dataset_name,
        offset=offset,
        limit=raw_limit,
        cache_dir=cache_dir,
        token=token,
    )):
        scanned += 1
        normalized = normalizer(row, source_offset=offset + relative_index)
        if normalized["id"] and normalized["text"]:
            if it_only and not looks_like_it(normalized, strict=it_strict):
                continue
            yielded += 1
            if yielded % 1000 == 0 or scanned % 5000 == 0:
                logging.info("%s: exported %s accepted rows after scanning %s raw rows", dataset_name, yielded, scanned)
            yield normalized
            if yielded >= limit:
                break

    if yielded < limit:
        logging.warning(
            "%s: requested %s rows but only found %s accepted rows after scanning %s raw rows starting at offset %s",
            dataset_name,
            limit,
            yielded,
            scanned,
            offset,
        )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if args.overwrite and args.resume:
        raise SystemExit("Use either --overwrite or --resume, not both.")
    if args.it_strict:
        args.it_only = True
    if args.output.exists() and not args.overwrite and not args.resume:
        raise SystemExit(f"{args.output} already exists. Pass --overwrite to replace it.")

    if args.output.exists() and args.overwrite:
        args.output.unlink()

    if not args.hf_token:
        logging.warning("HF_TOKEN is not set. Unauthenticated Hub requests can be slower and more rate-limited.")

    existing_cv = 0
    existing_jd = 0
    cv_resume_offset: int | None = None
    jd_resume_offset: int | None = None
    if args.resume and args.output.exists():
        existing_cv, existing_jd, cv_resume_offset, jd_resume_offset = inspect_existing_rows(args.output)
        if args.it_only:
            missing_resume_offsets = (existing_cv > 0 and cv_resume_offset is None) or (existing_jd > 0 and jd_resume_offset is None)
            if missing_resume_offsets:
                raise SystemExit(
                    "Cannot resume an IT-filtered output created without source_offset metadata. Re-run with --overwrite."
                )
        logging.info("Resuming from %s existing CV rows and %s existing JD rows in %s", existing_cv, existing_jd, args.output)

    cv_start_offset = cv_resume_offset if cv_resume_offset is not None else args.cv_offset + existing_cv
    jd_start_offset = jd_resume_offset if jd_resume_offset is not None else args.jd_offset + existing_jd
    cv_remaining = max(args.cv_limit - existing_cv, 0)
    jd_remaining = max(args.jd_limit - existing_jd, 0)
    cv_scan_budget = remaining_scan_budget(args.max_scan_cv, args.cv_offset, cv_start_offset)
    jd_scan_budget = remaining_scan_budget(args.max_scan_jd, args.jd_offset, jd_start_offset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv_written = existing_cv
    jd_written = existing_jd

    with args.output.open("a", encoding="utf-8") as handle:
        for row in iter_normalized_rows(
            dataset_name=CV_DATASET,
            offset=cv_start_offset,
            limit=cv_remaining,
            cache_dir=args.cache_dir,
            token=args.hf_token,
            normalizer=normalize_cv_row,
            it_only=args.it_only,
            it_strict=args.it_strict,
            max_scan=cv_scan_budget,
        ):
            write_row(handle, row)
            cv_written += 1

        for row in iter_normalized_rows(
            dataset_name=JD_DATASET,
            offset=jd_start_offset,
            limit=jd_remaining,
            cache_dir=args.cache_dir,
            token=args.hf_token,
            normalizer=normalize_jd_row,
            it_only=args.it_only,
            it_strict=args.it_strict,
            max_scan=jd_scan_budget,
        ):
            write_row(handle, row)
            jd_written += 1

    if cv_written < args.cv_limit or jd_written < args.jd_limit:
        raise SystemExit(
            f"Only collected {cv_written}/{args.cv_limit} CV rows and {jd_written}/{args.jd_limit} JD rows. "
            "Increase the scan window or remove restrictive filters."
        )

    logging.info("Finished writing %s CV rows and %s JD rows to %s", cv_written, jd_written, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
