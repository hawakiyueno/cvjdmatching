from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 90

ENTITY_LABELS = {
    "TECHNOLOGY",
    "JOB_ROLE",
    "SKILL",
    "WORK_ACTIVITY",
    "INDUSTRY",
    "PROJECT_TYPE",
    "DEGREE",
    "CERTIFICATION",
}

FACT_TYPES = {
    "EXPERIENCE_YEARS",
    "DEGREE",
    "CERTIFICATION",
}

ENTITY_LABEL_ALIASES = {
    "TECHNOLOGIES": "TECHNOLOGY",
    "TECH": "TECHNOLOGY",
    "TOOLS": "TECHNOLOGY",
    "TOOL": "TECHNOLOGY",
    "JOB": "JOB_ROLE",
    "JOBS": "JOB_ROLE",
    "JOBS_ROLE": "JOB_ROLE",
    "JOB_ROLES": "JOB_ROLE",
    "ROLE": "JOB_ROLE",
    "ROLES": "JOB_ROLE",
    "SKILLS": "SKILL",
    "WORK_ACTIVITIES": "WORK_ACTIVITY",
    "TASK": "WORK_ACTIVITY",
    "TASKS": "WORK_ACTIVITY",
    "INDUSTRIES": "INDUSTRY",
    "PROJECT_TYPES": "PROJECT_TYPE",
    "PROJECT": "PROJECT_TYPE",
    "PROJECTS": "PROJECT_TYPE",
    "EDUCATION": "DEGREE",
    "DEGREES": "DEGREE",
    "CERTIFICATIONS": "CERTIFICATION",
}

FACT_TYPE_ALIASES = {
    "EXPERIENCE": "EXPERIENCE_YEARS",
    "YEARS_EXPERIENCE": "EXPERIENCE_YEARS",
    "MIN_EXPERIENCE_YEARS": "EXPERIENCE_YEARS",
    "DEGREES": "DEGREE",
    "REQUIRED_DEGREE": "DEGREE",
    "CERTIFICATIONS": "CERTIFICATION",
    "REQUIRED_CERTIFICATION": "CERTIFICATION",
}

TEXT_COLUMN_CANDIDATES = (
    "text",
    "body",
    "content",
    "document",
    "document_text",
    "full_text",
    "raw_text",
    "cv",
    "cv_text",
    "resume",
    "resume_text",
    "profile",
    "profile_text",
    "jd",
    "jd_text",
    "job_description",
    "job_desc",
    "description",
    "vacancy_text",
    "posting_text",
)

ID_COLUMN_CANDIDATES = (
    "id",
    "_id",
    "uuid",
    "doc_id",
    "document_id",
    "record_id",
    "candidate_id",
    "job_id",
)

DOC_TYPE_COLUMN_CANDIDATES = (
    "doc_type",
    "document_type",
    "type",
    "source",
    "kind",
    "side",
    "record_type",
)

LANGUAGE_COLUMN_CANDIDATES = (
    "language",
    "lang",
)

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_language": {
            "type": "string",
            "description": "ISO-like language code such as en, vi, fr, or unknown.",
        },
        "is_it_document": {
            "type": "boolean",
            "description": "True only if the document is clearly about an IT job or IT candidate.",
        },
        "summary": {
            "type": "string",
            "description": "One sentence summary of the document content.",
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "One of TECHNOLOGY, JOB_ROLE, SKILL, WORK_ACTIVITY, INDUSTRY, PROJECT_TYPE, DEGREE, CERTIFICATION.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Exact span copied from the provided text chunk.",
                    },
                    "start": {
                        "type": "integer",
                        "description": "0-based inclusive start offset within the provided chunk only.",
                    },
                    "end": {
                        "type": "integer",
                        "description": "0-based exclusive end offset within the provided chunk only.",
                    },
                    "normalized": {
                        "type": "string",
                        "description": "Lowercase canonical form for mapping and deduplication.",
                    },
                },
                "required": ["label", "text", "start", "end", "normalized"],
                "additionalProperties": False,
            },
        },
        "qualification_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact_type": {
                        "type": "string",
                        "description": "One of EXPERIENCE_YEARS, DEGREE, CERTIFICATION.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Exact span copied from the provided text chunk.",
                    },
                    "start": {
                        "type": "integer",
                        "description": "0-based inclusive start offset within the provided chunk only.",
                    },
                    "end": {
                        "type": "integer",
                        "description": "0-based exclusive end offset within the provided chunk only.",
                    },
                    "operator": {
                        "type": "string",
                        "description": "Comparator such as >=, >, =, <=, preferred, or unknown.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Normalized scalar value, for example 3, bachelor, aws-certified-solutions-architect.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit such as years or credential.",
                    },
                    "normalized": {
                        "type": "string",
                        "description": "Lowercase canonical form for downstream matching.",
                    },
                    "is_mandatory": {
                        "type": "boolean",
                        "description": "True for explicit hard requirements, otherwise false.",
                    },
                },
                "required": [
                    "fact_type",
                    "text",
                    "start",
                    "end",
                    "operator",
                    "value",
                    "unit",
                    "normalized",
                    "is_mandatory",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "document_language",
        "is_it_document",
        "summary",
        "entities",
        "qualification_facts",
    ],
    "additionalProperties": False,
}

SYSTEM_INSTRUCTION = """You are creating weak labels for Stage 1 of a CV-JD matching pipeline.

The pipeline in the slide deck uses zero-shot LLM NER to bootstrap a high-quality dataset that will later train a custom span-based RoBERTa model. Your task is to read one document chunk from the Djinni dataset and return precise JSON only.

Follow these rules exactly:
1. Extract only spans that appear explicitly in the given text.
2. Offsets must use Python string indexing over the provided chunk only, with start inclusive and end exclusive.
3. The `text` field must be an exact copy of the source substring.
4. Overlapping entities are allowed when they are both valid.
5. Do not invent facts, infer hidden requirements, or paraphrase span text.
6. If the chunk is not clearly English, still return JSON but set `document_language` accordingly.
7. If the chunk is not clearly about an IT candidate or IT job, set `is_it_document` to false and keep arrays empty unless there are still obvious IT entities.
8. Focus on IT recruitment labels:
   - TECHNOLOGY: languages, frameworks, libraries, platforms, databases, cloud services, developer tools.
   - JOB_ROLE: job titles and role names.
   - SKILL: competencies that are not specific named technologies.
   - WORK_ACTIVITY: concrete work tasks or responsibilities, preferably full verb phrases instead of isolated verbs.
   - INDUSTRY: business domains like fintech or healthcare, not fields of study such as Computer Science.
   - PROJECT_TYPE: project categories like recommender systems, web applications, or computer vision projects.
   - DEGREE: educational credentials, preferably including the field when it is part of the same span.
   - CERTIFICATION: named certifications.
9. Extract qualification_facts for explicit experience duration, degree requirements or achievements, and certifications. For CVs, use operator `=` when the candidate states an achieved qualification or years of experience. For JDs, use the explicit comparator when present, otherwise `unknown`.
10. Mark `is_mandatory` true only for hard requirements such as must, required, mandatory, minimum, at least, or equivalent phrasing.
11. Return empty arrays when nothing relevant is present.
"""


@dataclass(frozen=True)
class TextSpec:
    column: str
    doc_type: str


class GeminiApiError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        body: str,
        *,
        retry_after_seconds: float | None = None,
        key_invalid: bool = False,
        quota_exhausted: bool = False,
    ) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body
        self.retry_after_seconds = retry_after_seconds
        self.key_invalid = key_invalid
        self.quota_exhausted = quota_exhausted


class AllGeminiKeysExhaustedError(RuntimeError):
    pass


def parse_retry_delay_seconds(value: str) -> float | None:
    stripped = value.strip().lower()
    if not stripped:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h)?", stripped)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    if unit == "ms":
        return amount / 1000.0
    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60.0
    if unit == "h":
        return amount * 3600.0
    return None


def build_gemini_api_error(status_code: int, body: str) -> GeminiApiError:
    retry_after_seconds = None
    key_invalid = status_code in {401, 403}
    quota_exhausted = False

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        error_payload = payload.get("error") or {}
        status_text = stringify(error_payload.get("status")).upper()
        message = stringify(error_payload.get("message")).lower()

        key_invalid = key_invalid or status_text in {"UNAUTHENTICATED", "PERMISSION_DENIED"}

        for detail in error_payload.get("details") or []:
            if not isinstance(detail, dict):
                continue
            retry_delay = stringify(detail.get("retryDelay"))
            parsed = parse_retry_delay_seconds(retry_delay)
            if parsed is not None:
                retry_after_seconds = parsed
            quota_id = stringify(detail.get("quotaId"))
            quota_metric = stringify(detail.get("quotaMetric")).lower()
            if "perday" in quota_id.lower() or "free_tier" in quota_metric:
                quota_exhausted = True

        if not quota_exhausted:
            quota_exhausted = (
                status_text == "RESOURCE_EXHAUSTED"
                and (
                    "current quota" in message
                    or "quota exceeded" in message
                    or "billing details" in message
                    or "free tier" in message
                )
            )
    elif status_code == 429:
        quota_exhausted = False

    return GeminiApiError(
        status_code,
        body,
        retry_after_seconds=retry_after_seconds,
        key_invalid=key_invalid,
        quota_exhausted=quota_exhausted,
    )


def merge_api_keys(candidates: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip()
        if not key or key in seen:
            continue
        merged.append(key)
        seen.add(key)
    return merged


def load_api_keys_from_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return merge_api_keys(handle.readlines())


class GeminiRestClient:
    def __init__(
        self,
        api_keys: list[str],
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 4,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        if not api_keys:
            raise ValueError("At least one Gemini API key is required.")
        self.api_keys = api_keys
        self.active_key_index = 0
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def generate_annotation(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseJsonSchema": RESPONSE_JSON_SCHEMA,
                "maxOutputTokens": 8192,
            },
        }

        encoded = json.dumps(payload).encode("utf-8")
        while True:
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_keys[self.active_key_index],
            }

            last_error: Exception | None = None
            rotated_key = False

            for attempt in range(1, self.max_retries + 1):
                req = request.Request(self.endpoint, data=encoded, headers=headers, method="POST")
                try:
                    with request.urlopen(req, timeout=self.timeout_seconds) as response:
                        raw_body = response.read().decode("utf-8")
                    raw = json.loads(raw_body)
                    text = extract_candidate_text(raw)
                    return parse_model_json(text), raw
                except error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    api_error = build_gemini_api_error(exc.code, body)
                    last_error = api_error

                    if api_error.key_invalid or api_error.quota_exhausted:
                        if self.rotate_api_key(api_error):
                            rotated_key = True
                            break
                        raise AllGeminiKeysExhaustedError(
                            "All Gemini API keys are exhausted or invalid. Rerun with --resume and a fresh key."
                        ) from api_error
                except Exception as exc:  # noqa: BLE001
                    last_error = exc

                if attempt < self.max_retries:
                    if isinstance(last_error, GeminiApiError) and last_error.retry_after_seconds is not None:
                        sleep_seconds = last_error.retry_after_seconds
                    else:
                        sleep_seconds = self.retry_backoff_seconds * attempt
                    logging.warning(
                        "Gemini call failed on attempt %s/%s; retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)

            if rotated_key:
                continue

            assert last_error is not None
            raise last_error

    def rotate_api_key(self, reason: GeminiApiError) -> bool:
        if self.active_key_index + 1 >= len(self.api_keys):
            return False
        previous_slot = self.active_key_index + 1
        self.active_key_index += 1
        next_slot = self.active_key_index + 1
        logging.warning(
            "Gemini API key slot %s is unavailable (%s). Switching to key slot %s/%s.",
            previous_slot,
            "quota exhausted" if reason.quota_exhausted else "invalid key",
            next_slot,
            len(self.api_keys),
        )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Djinni CV/JD text with Gemini zero-shot NER spans for downstream span-based NER training."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV or JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file for annotations.")
    parser.add_argument("--api-key", help="Single Gemini API key. Defaults to GEMINI_API_KEY if set.")
    parser.add_argument("--api-keys-file", type=Path, help="Text file with one Gemini API key per line. Keys are rotated on quota/auth failures.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--id-column", help="Column used as the stable source identifier.")
    parser.add_argument("--text-column", help="Single text column to annotate.")
    parser.add_argument(
        "--text-columns",
        help="Comma-separated text specs. Example: cv:resume_text,jd:job_description. If set, overrides --text-column.",
    )
    parser.add_argument(
        "--document-type",
        choices=("auto", "cv", "jd"),
        default="auto",
        help="Fallback document type when it cannot be inferred from metadata.",
    )
    parser.add_argument("--doc-type-column", help="Column storing document type/source metadata.")
    parser.add_argument("--language-column", help="Column storing language metadata.")
    parser.add_argument("--max-records", type=int, help="Stop after N source rows.")
    parser.add_argument("--max-documents", type=int, help="Stop after N emitted document annotations.")
    parser.add_argument("--max-chars-per-call", type=int, default=12000, help="Chunk long documents before calling Gemini.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap in characters between long text chunks.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between Gemini requests.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout per request.")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per Gemini call.")
    parser.add_argument("--require-english", action="store_true", help="Skip outputs whose inferred language is not English.")
    parser.add_argument("--require-it", action="store_true", help="Skip outputs whose inferred domain is not IT.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output JSONL by skipping seen ids.")
    parser.add_argument("--raw-output-dir", type=Path, help="Optional directory for raw Gemini responses.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def load_api_keys(args: argparse.Namespace) -> list[str]:
    keys: list[str] = []

    env_multi = os.environ.get("GEMINI_API_KEYS")
    if env_multi:
        keys.extend(re.split(r"[\r\n,]+", env_multi))

    env_single = os.environ.get("GEMINI_API_KEY")
    if env_single:
        keys.append(env_single)

    if args.api_keys_file:
        keys.extend(load_api_keys_from_file(args.api_keys_file))

    if args.api_key:
        keys.append(args.api_key)

    return merge_api_keys(keys)


def peek_fields(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or [])
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL line must be an object.")
                return list(row.keys())
        return []
    raise ValueError("Only CSV and JSONL inputs are supported.")


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield dict(row)
        return

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_number} in {path} is not a JSON object.")
                yield row
        return

    raise ValueError("Only CSV and JSONL inputs are supported.")


def match_field(fields: list[str], explicit: str | None, candidates: tuple[str, ...]) -> str | None:
    lowered = {field.lower(): field for field in fields}
    if explicit:
        if explicit in fields:
            return explicit
        lowered_name = explicit.lower()
        if lowered_name in lowered:
            return lowered[lowered_name]
        raise ValueError(f"Column not found: {explicit}")

    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def parse_text_specs(fields: list[str], args: argparse.Namespace) -> list[TextSpec]:
    if args.text_columns:
        specs: list[TextSpec] = []
        for raw_spec in args.text_columns.split(","):
            item = raw_spec.strip()
            if not item:
                continue
            doc_type = "auto"
            column = item
            if ":" in item:
                maybe_type, maybe_column = item.split(":", 1)
                maybe_type = maybe_type.strip().lower()
                maybe_column = maybe_column.strip()
                if maybe_type in {"cv", "jd"}:
                    doc_type = maybe_type
                    column = maybe_column
            resolved = match_field(fields, column, ())
            if not resolved:
                raise ValueError(f"Column not found in --text-columns: {column}")
            specs.append(TextSpec(column=resolved, doc_type=doc_type))
        if not specs:
            raise ValueError("No valid text columns were provided.")
        return specs

    text_column = match_field(fields, args.text_column, TEXT_COLUMN_CANDIDATES)
    if not text_column:
        raise ValueError(
            "Could not detect a text column automatically. Pass --text-column or --text-columns explicitly."
        )
    return [TextSpec(column=text_column, doc_type=args.document_type)]


def infer_doc_type(value: str | None, fallback: str, text_column: str) -> str:
    for candidate in (value or "", text_column, fallback):
        normalized = normalize_doc_type(candidate)
        if normalized != "unknown":
            return normalized
    return "unknown"


def normalize_doc_type(value: str | None) -> str:
    lowered = (value or "").strip().lower()
    if not lowered:
        return "unknown"
    if any(token in lowered for token in ("cv", "resume", "candidate", "profile")):
        return "cv"
    if any(token in lowered for token in ("jd", "job", "description", "vacancy", "posting", "role")):
        return "jd"
    return "unknown"


def normalize_language(value: str | None) -> str:
    lowered = (value or "").strip().lower()
    if not lowered:
        return "unknown"
    if lowered.startswith("en"):
        return "en"
    return lowered


def stable_record_id(row: dict[str, Any], id_column: str | None, row_index: int, text_value: str, source_column: str) -> str:
    if id_column:
        source_id = stringify(row.get(id_column))
        if source_id:
            return f"{source_id}:{source_column}"
    digest = hashlib.sha1(text_value.encode("utf-8")).hexdigest()[:12]
    return f"row-{row_index}:{source_column}:{digest}"


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def split_text(text: str, max_chars: int, overlap: int) -> list[tuple[int, str]]:
    if max_chars <= 0:
        raise ValueError("--max-chars-per-call must be positive.")
    if overlap < 0:
        raise ValueError("--chunk-overlap cannot be negative.")
    if len(text) <= max_chars:
        return [(0, text)]

    chunks: list[tuple[int, str]] = []
    cursor = 0
    while cursor < len(text):
        limit = min(cursor + max_chars, len(text))
        if limit < len(text):
            boundary = max(
                text.rfind("\n\n", cursor + max_chars // 2, limit),
                text.rfind("\n", cursor + max_chars // 2, limit),
                text.rfind(". ", cursor + max_chars // 2, limit),
                text.rfind(" ", cursor + max_chars // 2, limit),
            )
            if boundary > cursor:
                limit = boundary + 1

        raw = text[cursor:limit]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        chunk_start = cursor + leading
        chunk_end = limit - trailing
        if chunk_end <= chunk_start:
            chunk_start = cursor
            chunk_end = limit

        chunks.append((chunk_start, text[chunk_start:chunk_end]))
        if chunk_end >= len(text):
            break
        cursor = max(chunk_end - overlap, cursor + 1)
    return chunks


def build_prompt(record_id: str, doc_type: str, chunk_index: int, chunk_start: int, chunk_text: str) -> str:
    return f"""Annotate this Djinni document chunk.

record_id: {record_id}
document_type: {doc_type}
chunk_index: {chunk_index}
chunk_start_in_full_document: {chunk_start}

Return JSON that matches the provided schema. Offsets must be relative to the chunk text below, not the full document.

Chunk text:
<<<TEXT
{chunk_text}
TEXT
"""


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def extract_candidate_text(raw: dict[str, Any]) -> str:
    candidates = raw.get("candidates") or []
    if not candidates:
        raise ValueError(f"No candidates returned by Gemini: {json.dumps(raw)[:500]}")

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    if finish_reason and finish_reason != "STOP":
        raise ValueError(f"Gemini finishReason={finish_reason}: {json.dumps(candidate)[:500]}")

    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    joined = "".join(texts).strip()
    if not joined:
        raise ValueError(f"Gemini returned empty content: {json.dumps(raw)[:500]}")
    return joined


def normalize_entity_label(value: str) -> str | None:
    candidate = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    candidate = ENTITY_LABEL_ALIASES.get(candidate, candidate)
    if candidate in ENTITY_LABELS:
        return candidate
    return None


def normalize_fact_type(value: str) -> str | None:
    candidate = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    candidate = FACT_TYPE_ALIASES.get(candidate, candidate)
    if candidate in FACT_TYPES:
        return candidate
    return None


def collapse_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def canonicalize_operator(raw_operator: str, fact_type: str, is_mandatory: bool) -> str:
    value = raw_operator.strip().lower()
    if value in {"=", "=="}:
        return "="
    if value in {">", "<", ">=", "<="}:
        return value
    if "preferred" in value or "nice to have" in value:
        return "preferred"
    if "at least" in value or "minimum" in value or "min " in value or "or more" in value:
        return ">="
    if "less than" in value:
        return "<"
    if "no more than" in value or "at most" in value:
        return "<="
    if value in {"required", "must", "mandatory"}:
        return ">=" if fact_type == "EXPERIENCE_YEARS" else "unknown"
    if not is_mandatory and fact_type == "EXPERIENCE_YEARS":
        return "="
    if not is_mandatory and fact_type in {"DEGREE", "CERTIFICATION"}:
        return "="
    return "unknown"


def repair_offsets(text: str, span_text: str, start: int | None, end: int | None) -> tuple[int, int] | None:
    if not span_text:
        return None
    if start is not None and end is not None and 0 <= start <= end <= len(text):
        if text[start:end] == span_text:
            return (start, end)

    matches = [match for match in re.finditer(re.escape(span_text), text)]
    if not matches:
        return None

    if start is None:
        match = matches[0]
        return (match.start(), match.end())

    match = min(matches, key=lambda item: abs(item.start() - start))
    return (match.start(), match.end())


def sanitize_entity(text: str, entity: dict[str, Any], shift: int) -> dict[str, Any] | None:
    label = normalize_entity_label(stringify(entity.get("label")))
    if not label:
        return None
    span_text = stringify(entity.get("text"))
    offsets = repair_offsets(
        text,
        span_text,
        to_int(entity.get("start")),
        to_int(entity.get("end")),
    )
    if not offsets:
        return None
    start, end = offsets
    normalized = stringify(entity.get("normalized")) or collapse_space(span_text)
    return {
        "label": label,
        "text": span_text,
        "start": start + shift,
        "end": end + shift,
        "normalized": normalized,
    }


def sanitize_fact(text: str, fact: dict[str, Any], shift: int) -> dict[str, Any] | None:
    fact_type = normalize_fact_type(stringify(fact.get("fact_type")))
    if not fact_type:
        return None
    span_text = stringify(fact.get("text"))
    offsets = repair_offsets(
        text,
        span_text,
        to_int(fact.get("start")),
        to_int(fact.get("end")),
    )
    if not offsets:
        return None
    start, end = offsets
    is_mandatory = bool(fact.get("is_mandatory", False))
    operator = canonicalize_operator(stringify(fact.get("operator")), fact_type, is_mandatory)
    value = stringify(fact.get("value"))
    unit = stringify(fact.get("unit"))
    normalized = stringify(fact.get("normalized")) or collapse_space(span_text)
    return {
        "fact_type": fact_type,
        "text": span_text,
        "start": start + shift,
        "end": end + shift,
        "operator": operator,
        "value": value,
        "unit": unit,
        "normalized": normalized,
        "is_mandatory": is_mandatory,
    }


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entity in entities:
        key = (entity["label"], entity["start"], entity["end"], entity["text"])
        unique[key] = entity
    return sorted(unique.values(), key=lambda item: (item["start"], item["end"], item["label"]))


def dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for fact in facts:
        key = (
            fact["fact_type"],
            fact["start"],
            fact["end"],
            fact["text"],
            fact["operator"],
            fact["value"],
        )
        unique[key] = fact
    return sorted(unique.values(), key=lambda item: (item["start"], item["end"], item["fact_type"]))


def sanitize_annotation(
    full_text: str,
    chunks: list[tuple[int, str]],
    chunk_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    language_votes: list[str] = []
    it_votes: list[bool] = []
    summary_parts: list[str] = []

    for (chunk_start, chunk_text), raw_output in zip(chunks, chunk_outputs, strict=True):
        language_votes.append(normalize_language(stringify(raw_output.get("document_language"))))
        it_votes.append(bool(raw_output.get("is_it_document", False)))
        summary = stringify(raw_output.get("summary"))
        if summary:
            summary_parts.append(summary)

        for entity in raw_output.get("entities") or []:
            if isinstance(entity, dict):
                cleaned = sanitize_entity(chunk_text, entity, chunk_start)
                if cleaned:
                    entities.append(cleaned)

        for fact in raw_output.get("qualification_facts") or []:
            if isinstance(fact, dict):
                cleaned = sanitize_fact(chunk_text, fact, chunk_start)
                if cleaned:
                    facts.append(cleaned)

    document_language = most_common(language_votes, default="unknown")
    summary = " ".join(summary_parts[:2]).strip()
    if len(summary) > 400:
        summary = summary[:397].rstrip() + "..."

    return {
        "document_language": document_language,
        "is_it_document": any(it_votes),
        "summary": summary,
        "entities": dedupe_entities(entities),
        "qualification_facts": dedupe_facts(facts),
        "text_length": len(full_text),
    }


def most_common(values: list[Any], default: Any) -> Any:
    counts: dict[Any, int] = {}
    for value in values:
        if value in ("", None, "unknown"):
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return default
    return max(counts.items(), key=lambda item: item[1])[0]


def load_processed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    processed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            record_id = stringify(row.get("record_id"))
            if record_id:
                processed.add(record_id)
    return processed


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def error_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.errors.jsonl")


def maybe_save_raw_response(raw_output_dir: Path | None, record_id: str, chunk_index: int, raw: dict[str, Any]) -> None:
    if raw_output_dir is None:
        return
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    safe_record_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id)
    target = raw_output_dir / f"{safe_record_id}.chunk-{chunk_index}.json"
    target.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def annotate_document(
    client: GeminiRestClient,
    record_id: str,
    doc_type: str,
    text: str,
    max_chars_per_call: int,
    chunk_overlap: int,
    sleep_seconds: float,
    raw_output_dir: Path | None,
) -> dict[str, Any]:
    chunks = split_text(text, max_chars_per_call, chunk_overlap)
    chunk_outputs: list[dict[str, Any]] = []

    for chunk_index, (chunk_start, chunk_text) in enumerate(chunks):
        prompt = build_prompt(record_id, doc_type, chunk_index, chunk_start, chunk_text)
        annotation, raw = client.generate_annotation(prompt)
        chunk_outputs.append(annotation)
        maybe_save_raw_response(raw_output_dir, record_id, chunk_index, raw)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return sanitize_annotation(text, chunks, chunk_outputs)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    api_keys = load_api_keys(args)
    if not api_keys:
        raise SystemExit("Gemini API key missing. Pass --api-key, --api-keys-file, or set GEMINI_API_KEY / GEMINI_API_KEYS.")
    logging.info("Loaded %s Gemini API key(s).", len(api_keys))

    fields = peek_fields(args.input)
    if not fields:
        raise SystemExit(f"No rows found in {args.input}.")

    id_column = match_field(fields, args.id_column, ID_COLUMN_CANDIDATES)
    doc_type_column = match_field(fields, args.doc_type_column, DOC_TYPE_COLUMN_CANDIDATES)
    language_column = match_field(fields, args.language_column, LANGUAGE_COLUMN_CANDIDATES)
    text_specs = parse_text_specs(fields, args)

    if args.resume:
        processed_ids = load_processed_ids(args.output)
        logging.info("Loaded %s processed ids from %s", len(processed_ids), args.output)
    else:
        processed_ids = set()

    client = GeminiRestClient(
        api_keys=api_keys,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    emitted_documents = 0
    visited_rows = 0

    for row_index, row in enumerate(iter_rows(args.input), start=1):
        visited_rows += 1
        if args.max_records and visited_rows > args.max_records:
            break

        row_doc_type_value = stringify(row.get(doc_type_column)) if doc_type_column else ""
        row_language_value = stringify(row.get(language_column)) if language_column else ""

        for text_spec in text_specs:
            text_value = stringify(row.get(text_spec.column))
            if not text_value:
                continue

            record_id = stable_record_id(row, id_column, row_index, text_value, text_spec.column)
            if record_id in processed_ids:
                continue

            doc_type = infer_doc_type(row_doc_type_value, text_spec.doc_type, text_spec.column)
            source_language = normalize_language(row_language_value)
            logging.info("Annotating %s (%s, %s chars)", record_id, doc_type, len(text_value))

            try:
                annotation = annotate_document(
                    client=client,
                    record_id=record_id,
                    doc_type=doc_type,
                    text=text_value,
                    max_chars_per_call=args.max_chars_per_call,
                    chunk_overlap=args.chunk_overlap,
                    sleep_seconds=args.sleep_seconds,
                    raw_output_dir=args.raw_output_dir,
                )
            except AllGeminiKeysExhaustedError as exc:
                write_jsonl(
                    error_output_path(args.output),
                    {
                        "record_id": record_id,
                        "row_index": row_index,
                        "source_column": text_spec.column,
                        "error": str(exc),
                        "fatal": True,
                    },
                )
                logging.error("%s", exc)
                logging.error(
                    "Progress is saved in %s. Add a new Gemini key and rerun the same command with --resume.",
                    args.output,
                )
                return 2
            except Exception as exc:  # noqa: BLE001
                write_jsonl(
                    error_output_path(args.output),
                    {
                        "record_id": record_id,
                        "row_index": row_index,
                        "source_column": text_spec.column,
                        "error": str(exc),
                    },
                )
                logging.exception("Failed to annotate %s", record_id)
                continue

            if args.require_english and annotation["document_language"] != "en":
                logging.info("Skipping %s because inferred language is %s", record_id, annotation["document_language"])
                continue
            if args.require_it and not annotation["is_it_document"]:
                logging.info("Skipping %s because the document is not classified as IT", record_id)
                continue

            output_row = {
                "record_id": record_id,
                "source_path": str(args.input),
                "source_row_index": row_index,
                "source_column": text_spec.column,
                "source_language": source_language,
                "document_type": doc_type,
                "model": args.model,
                "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
                "text": text_value,
                "document_language": annotation["document_language"],
                "is_it_document": annotation["is_it_document"],
                "summary": annotation["summary"],
                "entities": annotation["entities"],
                "qualification_facts": annotation["qualification_facts"],
                "text_length": annotation["text_length"],
            }
            write_jsonl(args.output, output_row)
            processed_ids.add(record_id)
            emitted_documents += 1

            if args.max_documents and emitted_documents >= args.max_documents:
                logging.info("Reached --max-documents=%s", args.max_documents)
                return 0

    logging.info("Finished. Visited %s source rows and emitted %s documents.", visited_rows, emitted_documents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
