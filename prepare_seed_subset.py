from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from ftfy import fix_text as ftfy_fix_text
except ImportError:  # pragma: no cover - optional dependency
    ftfy_fix_text = None


MANUAL_TEXT_REPLACEMENTS = {
    "\u00a0": " ",
    "\u200b": "",
    "\ufeff": "",
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\x9d": '"',
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u00a6": "...",
    "\u00e2\u20ac\u00a2": "-",
    "\u2022": "-",
    "\u00c2 ": " ",
    "\u00c2": "",
    "\u0110\u00a1#": "C#",
}

CV_BUCKETS = (
    (400, "short"),
    (900, "medium"),
    (1800, "long"),
)

JD_BUCKETS = (
    (1000, "short"),
    (1800, "medium"),
    (3000, "long"),
)

IT_PRIMARY_KEYWORDS = {
    "Android",
    "QA Automation",
    "JavaScript",
    "Node.js",
    "PHP",
    "Python",
    ".NET",
    "Unity",
    "Java",
    "Data Analyst",
    "Data Engineer",
    "Data Science",
    "Golang",
    "SQL",
    "DevOps",
    "QA",
    "Security",
    "Ruby",
    "Sysadmin",
    "C++",
    "Scala",
    "Rust",
    "Flutter",
    "iOS",
    "Salesforce",
}

AMBIGUOUS_PRIMARY_KEYWORDS = {
    "Other",
    "Lead",
    "Project Manager",
    "Support",
    "Business Analyst",
    "Product Manager",
    "Scrum Master",
    "Technical Writing",
}

STRICT_IT_PRIMARY_KEYWORDS = {
    "Android",
    "QA Automation",
    "JavaScript",
    "Node.js",
    "PHP",
    "Python",
    ".NET",
    "Unity",
    "Java",
    "Data Analyst",
    "Data Engineer",
    "Data Science",
    "Golang",
    "SQL",
    "DevOps",
    "QA",
    "Security",
    "Ruby",
    "Sysadmin",
    "C++",
    "Scala",
    "Rust",
    "Flutter",
    "iOS",
    "Salesforce",
}

IT_SIGNAL_TERMS = (
    "developer",
    "engineer",
    "devops",
    "data scientist",
    "data engineer",
    "data analyst",
    "qa",
    "automation",
    "tester",
    "test engineer",
    "backend",
    "back-end",
    "frontend",
    "front-end",
    "fullstack",
    "full stack",
    "android",
    "ios",
    "mobile app",
    "java",
    "javascript",
    "typescript",
    "python",
    ".net",
    "c#",
    "c++",
    "php",
    "node",
    "react",
    "angular",
    "golang",
    "go developer",
    "ruby",
    "scala",
    "rust",
    "flutter",
    "sql",
    "database",
    "aws",
    "azure",
    "gcp",
    "cloud",
    "docker",
    "kubernetes",
    "linux",
    "sysadmin",
    "system administrator",
    "network engineer",
    "security",
    "cybersecurity",
    "machine learning",
    "artificial intelligence",
    "blockchain",
    "web3",
    "unity",
    "unreal",
    "software",
    "technical writer",
    "salesforce",
)

STRICT_IT_ROLE_TERMS = (
    "software engineer",
    "software developer",
    "backend developer",
    "backend engineer",
    "frontend developer",
    "frontend engineer",
    "front-end developer",
    "front-end engineer",
    "fullstack developer",
    "fullstack engineer",
    "full stack developer",
    "full stack engineer",
    "android developer",
    "android engineer",
    "ios developer",
    "ios engineer",
    "mobile developer",
    "mobile engineer",
    "devops engineer",
    "platform engineer",
    "site reliability engineer",
    "sre",
    "qa engineer",
    "qa automation",
    "test automation engineer",
    "automation engineer",
    "data engineer",
    "data scientist",
    "data analyst",
    "machine learning engineer",
    "ml engineer",
    "ai engineer",
    "security engineer",
    "cybersecurity engineer",
    "cloud engineer",
    "system administrator",
    "sysadmin",
    "database administrator",
    "dba",
    "network engineer",
    "embedded engineer",
    "unity developer",
    "game developer",
    "abap developer",
    "sap developer",
    "salesforce developer",
    "1c developer",
    "developer",
    "engineer",
)

STRICT_IT_TECH_TERMS = (
    "python",
    "java",
    "javascript",
    "typescript",
    "node.js",
    "node ",
    "react",
    "angular",
    "vue",
    ".net",
    "c#",
    "c++",
    "php",
    "golang",
    "go ",
    "ruby",
    "scala",
    "rust",
    "sql",
    "postgres",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "linux",
    "airflow",
    "spark",
    "etl",
    "api",
    "microservice",
    "backend",
    "frontend",
    "android",
    "ios",
    "flutter",
    "unity",
    "unreal",
    "machine learning",
    "deep learning",
    "computer vision",
    "nlp",
    "blockchain",
    "web3",
    "salesforce",
    "sap",
    "abap",
    "1c",
    "database",
    "kafka",
    "terraform",
    "ci/cd",
)

STRICT_NON_IT_HEADER_TERMS = (
    "business analyst",
    "product manager",
    "project manager",
    "scrum master",
    "support",
    "technical writer",
    "technical writing",
    "copywriter",
    "marketing",
    "sales",
    "recruiter",
    "human resources",
    "hr ",
    "artist",
    "designer",
    "accountant",
    "finance",
    "financial",
    "lawyer",
    "legal",
    "teacher",
    "translator",
    "account manager",
    "customer service",
    "seo",
    "content",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter, clean, and sample a balanced seed subset from the 20k recruitment JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL, typically hf_recruitment_subset_20k.jsonl.")
    parser.add_argument("--seed-output", required=True, type=Path, help="Output JSONL for the cleaned seed subset.")
    parser.add_argument("--cleaned-output", type=Path, help="Optional output JSONL for the full cleaned dataset after filtering.")
    parser.add_argument("--stats-output", type=Path, help="Optional JSON summary of filtering and sampling stats.")
    parser.add_argument("--seed-cv", type=int, default=500, help="Number of CV rows in the seed subset.")
    parser.add_argument("--seed-jd", type=int, default=500, help="Number of JD rows in the seed subset.")
    parser.add_argument("--min-cv-chars", type=int, default=200, help="Minimum cleaned character length for CV rows.")
    parser.add_argument("--min-jd-chars", type=int, default=350, help="Minimum cleaned character length for JD rows.")
    parser.add_argument("--min-cv-words", type=int, default=30, help="Minimum word count for CV rows.")
    parser.add_argument("--min-jd-words", type=int, default=50, help="Minimum word count for JD rows.")
    parser.add_argument("--it-only", action="store_true", help="Keep only records that look like IT roles.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they already exist.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def clean_text(value: str) -> str:
    text = value
    if ftfy_fix_text is not None:
        text = ftfy_fix_text(text)

    for source, target in MANUAL_TEXT_REPLACEMENTS.items():
        text = text.replace(source, target)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w+#./-]+\b", text))


def dedupe_key(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def length_bucket(doc_type: str, text_length: int) -> str:
    buckets = CV_BUCKETS if doc_type == "cv" else JD_BUCKETS
    for upper_bound, label in buckets:
        if text_length < upper_bound:
            return label
    return "xlong"


def build_it_probe(row: dict[str, Any], cleaned_text: str) -> tuple[str, str]:
    primary_keyword = str(row.get("primary_keyword") or "").strip()
    position = str(row.get("position") or "").strip()
    header_probe = " ".join([primary_keyword, position]).lower()
    body_probe = " ".join([header_probe, cleaned_text[:1500]]).lower()
    return header_probe, body_probe


def is_it_row(row: dict[str, Any], cleaned_text: str, strict: bool = False) -> bool:
    primary_keyword = str(row.get("primary_keyword") or "").strip()
    header_probe, body_probe = build_it_probe(row, cleaned_text)

    if strict:
        if primary_keyword in STRICT_IT_PRIMARY_KEYWORDS:
            return True
        if any(term in header_probe for term in STRICT_NON_IT_HEADER_TERMS):
            return False
        has_strong_role = any(term in header_probe for term in STRICT_IT_ROLE_TERMS)
        if not has_strong_role:
            return False
        has_strong_tech = any(term in body_probe for term in STRICT_IT_TECH_TERMS)
        return has_strong_tech

    if primary_keyword in IT_PRIMARY_KEYWORDS:
        return True
    if primary_keyword in AMBIGUOUS_PRIMARY_KEYWORDS:
        return any(term in body_probe for term in IT_SIGNAL_TERMS)
    return False


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} already exists. Pass --overwrite to replace it.")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_and_clean_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "input_rows": 0,
        "kept_rows": 0,
        "dropped_missing_core_fields": 0,
        "dropped_language": 0,
        "dropped_non_it": 0,
        "dropped_short_or_sparse": 0,
        "dropped_unknown_type": 0,
        "dropped_duplicate_text": 0,
        "kept_by_type": Counter(),
    }
    kept_rows: list[dict[str, Any]] = []
    seen_by_type: dict[str, set[str]] = {"cv": set(), "jd": set()}

    with args.input.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            stats["input_rows"] += 1

            row_id = str(row.get("id", "")).strip()
            doc_type = str(row.get("doc_type", "")).strip().lower()
            source_language = str(row.get("source_language", "")).strip().lower()
            raw_text = str(row.get("text", "")).strip()

            if not row_id or not raw_text:
                stats["dropped_missing_core_fields"] += 1
                continue
            if doc_type not in {"cv", "jd"}:
                stats["dropped_unknown_type"] += 1
                continue
            if source_language and not source_language.startswith("en"):
                stats["dropped_language"] += 1
                continue

            cleaned = clean_text(raw_text)
            words = count_words(cleaned)
            chars = len(cleaned)
            min_chars = args.min_cv_chars if doc_type == "cv" else args.min_jd_chars
            min_words = args.min_cv_words if doc_type == "cv" else args.min_jd_words

            if args.it_only and not is_it_row(row, cleaned):
                stats["dropped_non_it"] += 1
                continue

            if chars < min_chars or words < min_words:
                stats["dropped_short_or_sparse"] += 1
                continue

            normalized = dedupe_key(cleaned)
            if normalized in seen_by_type[doc_type]:
                stats["dropped_duplicate_text"] += 1
                continue
            seen_by_type[doc_type].add(normalized)

            cleaned_row = dict(row)
            cleaned_row["text"] = cleaned
            cleaned_row["text_length"] = chars
            cleaned_row["word_count"] = words
            cleaned_row["length_bucket"] = length_bucket(doc_type, chars)
            cleaned_row["seed_group"] = f"{cleaned_row.get('primary_keyword') or 'unknown'}::{cleaned_row['length_bucket']}"

            kept_rows.append(cleaned_row)
            stats["kept_rows"] += 1
            stats["kept_by_type"][doc_type] += 1

    stats["kept_by_type"] = dict(stats["kept_by_type"])
    return kept_rows, stats


def select_diverse_rows(rows: list[dict[str, Any]], target_count: int, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("primary_keyword") or "unknown"), str(row.get("length_bucket") or "unknown"))
        grouped[key].append(row)

    active_keys = sorted(grouped)
    for key in active_keys:
        grouped[key].sort(key=lambda item: str(item.get("id")))
        rng.shuffle(grouped[key])

    selected: list[dict[str, Any]] = []
    while len(selected) < target_count and active_keys:
        next_round: list[tuple[str, str]] = []
        for key in active_keys:
            bucket = grouped[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if bucket:
                next_round.append(key)
            if len(selected) >= target_count:
                break
        active_keys = next_round
    return selected


def build_seed_subset(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cv_rows = [row for row in rows if row["doc_type"] == "cv"]
    jd_rows = [row for row in rows if row["doc_type"] == "jd"]

    if len(cv_rows) < args.seed_cv:
        raise SystemExit(f"Not enough cleaned CV rows for seed selection: need {args.seed_cv}, have {len(cv_rows)}.")
    if len(jd_rows) < args.seed_jd:
        raise SystemExit(f"Not enough cleaned JD rows for seed selection: need {args.seed_jd}, have {len(jd_rows)}.")

    selected_cv = select_diverse_rows(cv_rows, args.seed_cv, random.Random(args.random_seed))
    selected_jd = select_diverse_rows(jd_rows, args.seed_jd, random.Random(args.random_seed + 1))

    seed_rows = sorted(selected_cv + selected_jd, key=lambda item: (item["doc_type"], item["id"]))
    seed_stats = {
        "seed_rows": len(seed_rows),
        "seed_by_type": dict(Counter(row["doc_type"] for row in seed_rows)),
        "seed_by_keyword_top20": dict(Counter(str(row.get("primary_keyword") or "unknown") for row in seed_rows).most_common(20)),
        "seed_by_bucket": dict(Counter(str(row.get("length_bucket") or "unknown") for row in seed_rows)),
    }
    return seed_rows, seed_stats


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    ensure_writable(args.seed_output, args.overwrite)
    if args.cleaned_output:
        ensure_writable(args.cleaned_output, args.overwrite)
    if args.stats_output:
        ensure_writable(args.stats_output, args.overwrite)

    cleaned_rows, cleaning_stats = load_and_clean_rows(args)
    logging.info("Kept %s/%s rows after cleaning/filtering.", cleaning_stats["kept_rows"], cleaning_stats["input_rows"])

    seed_rows, seed_stats = build_seed_subset(cleaned_rows, args)
    write_jsonl(args.seed_output, seed_rows)
    logging.info("Wrote %s seed rows to %s", len(seed_rows), args.seed_output)

    if args.cleaned_output:
        write_jsonl(args.cleaned_output, cleaned_rows)
        logging.info("Wrote %s cleaned rows to %s", len(cleaned_rows), args.cleaned_output)

    combined_stats = {
        "cleaning": cleaning_stats,
        "seed": seed_stats,
    }
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(json.dumps(combined_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Seed by type: %s", seed_stats["seed_by_type"])
    logging.info("Seed by bucket: %s", seed_stats["seed_by_bucket"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
