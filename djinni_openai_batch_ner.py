from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from djinni_gemini_ner import (
    DEFAULT_TIMEOUT_SECONDS,
    DOC_TYPE_COLUMN_CANDIDATES,
    ID_COLUMN_CANDIDATES,
    LANGUAGE_COLUMN_CANDIDATES,
    RESPONSE_JSON_SCHEMA,
    configure_logging,
    error_output_path,
    infer_doc_type,
    iter_rows,
    load_processed_ids,
    match_field,
    normalize_language,
    parse_text_specs,
    peek_fields,
    sanitize_annotation,
    split_text,
    stable_record_id,
    stringify,
    write_jsonl,
)
from djinni_openai_ner import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    PROMPT_VERSION,
    RESPONSE_SCHEMA_NAME,
    SYSTEM_INSTRUCTION,
    build_prompt,
    extract_message_text,
    load_api_keys,
    parse_model_json,
    refine_annotation,
)


DEFAULT_BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_MAX_FILE_BYTES = 150 * 1024 * 1024
DEFAULT_MAX_REQUESTS_PER_SHARD = 25_000
DEFAULT_MAX_ESTIMATED_TOKENS_PER_SHARD = 1_500_000
DEFAULT_TRANSIENT_RETRIES = 6


@dataclass
class PreparedRecord:
    metadata: dict[str, Any]
    chunk_rows: list[dict[str, Any]]
    request_lines: list[bytes]
    total_bytes: int
    estimated_tokens: int


class OpenAIBatchError(RuntimeError):
    pass


def parse_retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def is_transient_http_status(status_code: int) -> bool:
    return status_code == 429 or status_code in {500, 502, 503, 504}


def retry_delay_seconds(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None and retry_after > 0:
        return retry_after
    return min(30.0, 2.0 * attempt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API pipeline for Stage 1 CV/JD weak-label NER generation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare batch request shards and local manifests.")
    prepare.add_argument("--input", required=True, type=Path, help="Input CSV or JSONL file.")
    prepare.add_argument("--work-dir", required=True, type=Path, help="Output working directory for manifests, requests, and outputs.")
    prepare.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model name. Default: {DEFAULT_MODEL}")
    prepare.add_argument("--id-column", help="Column used as the stable source identifier.")
    prepare.add_argument("--text-column", help="Single text column to annotate.")
    prepare.add_argument(
        "--text-columns",
        help="Comma-separated text specs. Example: cv:resume_text,jd:job_description. If set, overrides --text-column.",
    )
    prepare.add_argument(
        "--document-type",
        choices=("auto", "cv", "jd"),
        default="auto",
        help="Fallback document type when it cannot be inferred from metadata.",
    )
    prepare.add_argument("--doc-type-column", help="Column storing document type/source metadata.")
    prepare.add_argument("--language-column", help="Column storing language metadata.")
    prepare.add_argument("--max-records", type=int, help="Stop after N source rows.")
    prepare.add_argument("--max-documents", type=int, help="Stop after N prepared document annotations.")
    prepare.add_argument("--max-chars-per-call", type=int, default=12000, help="Chunk long documents before generating batch requests.")
    prepare.add_argument("--chunk-overlap", type=int, default=200, help="Overlap in characters between long text chunks.")
    prepare.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES, help="Maximum batch input file size per shard.")
    prepare.add_argument(
        "--max-requests-per-shard",
        type=int,
        default=DEFAULT_MAX_REQUESTS_PER_SHARD,
        help="Maximum number of request lines per shard.",
    )
    prepare.add_argument(
        "--max-estimated-tokens-per-shard",
        type=int,
        default=DEFAULT_MAX_ESTIMATED_TOKENS_PER_SHARD,
        help="Conservative token budget per shard used to stay below Batch API enqueued token limits.",
    )
    prepare.add_argument("--overwrite", action="store_true", help="Overwrite the work directory metadata and request shards.")
    prepare.add_argument(
        "--resume-output",
        type=Path,
        help="Optional existing final annotation JSONL. Prepared records already in that file are skipped.",
    )
    prepare.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    submit = subparsers.add_parser("submit", help="Upload request shards and create OpenAI batch jobs.")
    submit.add_argument("--work-dir", required=True, type=Path, help="Existing batch work directory created by prepare.")
    submit.add_argument("--api-key", help="Single OpenAI API key. Defaults to OPENAI_API_KEY if set.")
    submit.add_argument("--api-keys-file", type=Path, help="Text file with one OpenAI API key per line. The first key is used.")
    submit.add_argument("--completion-window", default=DEFAULT_COMPLETION_WINDOW, help=f"Batch completion window. Default: {DEFAULT_COMPLETION_WINDOW}")
    submit.add_argument("--shard-id", help="Optional single shard id to submit, for example shard-00001.")
    submit.add_argument("--max-shards", type=int, default=1, help="Maximum number of unsubmitted shards to submit in this run. Default: 1")
    submit.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout per request.")
    submit.add_argument("--overwrite", action="store_true", help="Replace existing jobs metadata and resubmit all shards.")
    submit.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    retarget = subparsers.add_parser(
        "retarget-model",
        help="Rewrite only unsubmitted shards in an existing work directory to a different OpenAI model.",
    )
    retarget.add_argument("--work-dir", required=True, type=Path, help="Existing batch work directory created by prepare.")
    retarget.add_argument("--model", required=True, help="New OpenAI model for shards that have not been submitted yet.")
    retarget.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    status = subparsers.add_parser("status", help="Fetch and save latest status for all submitted batch jobs.")
    status.add_argument("--work-dir", required=True, type=Path, help="Existing batch work directory.")
    status.add_argument("--api-key", help="Single OpenAI API key. Defaults to OPENAI_API_KEY if set.")
    status.add_argument("--api-keys-file", type=Path, help="Text file with one OpenAI API key per line. The first key is used.")
    status.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout per request.")
    status.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    download = subparsers.add_parser("download", help="Download completed batch output and error files.")
    download.add_argument("--work-dir", required=True, type=Path, help="Existing batch work directory.")
    download.add_argument("--api-key", help="Single OpenAI API key. Defaults to OPENAI_API_KEY if set.")
    download.add_argument("--api-keys-file", type=Path, help="Text file with one OpenAI API key per line. The first key is used.")
    download.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout per request.")
    download.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    finalize = subparsers.add_parser("finalize", help="Merge batch outputs into the final annotation JSONL.")
    finalize.add_argument("--work-dir", required=True, type=Path, help="Existing batch work directory.")
    finalize.add_argument("--output", required=True, type=Path, help="Final output JSONL file for merged annotations.")
    finalize.add_argument("--resume", action="store_true", help="Resume from an existing output JSONL by skipping seen ids.")
    finalize.add_argument("--require-english", action="store_true", help="Skip outputs whose inferred language is not English.")
    finalize.add_argument("--require-it", action="store_true", help="Skip outputs whose inferred domain is not IT.")
    finalize.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    return parser.parse_args()


def manifest_path(work_dir: Path) -> Path:
    return work_dir / "manifest.json"


def records_path(work_dir: Path) -> Path:
    return work_dir / "records.jsonl"


def jobs_path(work_dir: Path) -> Path:
    return work_dir / "jobs.json"


def requests_dir(work_dir: Path) -> Path:
    return work_dir / "requests"


def chunks_dir(work_dir: Path) -> Path:
    return work_dir / "chunks"


def outputs_dir(work_dir: Path) -> Path:
    return work_dir / "outputs"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def custom_chunk_id(record_id: str, chunk_index: int, chunk_start: int) -> str:
    digest = hashlib.sha1(f"{record_id}|{chunk_index}|{chunk_start}".encode("utf-8")).hexdigest()[:16]
    return f"chunk-{digest}"


def build_request_body(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": RESPONSE_SCHEMA_NAME,
                "strict": True,
                "schema": RESPONSE_JSON_SCHEMA,
            },
        },
    }


def build_batch_request_line(custom_id: str, model: str, prompt: str) -> bytes:
    row = {
        "custom_id": custom_id,
        "method": "POST",
        "url": DEFAULT_BATCH_ENDPOINT,
        "body": build_request_body(model, prompt),
    }
    return json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"


def estimate_tokens_from_request_line(line: bytes) -> int:
    # Conservative heuristic based on serialized JSON size.
    # This intentionally overestimates a bit so shard sizes stay under org queue limits.
    return max(1, (len(line) + 3) // 4)


def prepare_record(
    *,
    row: dict[str, Any],
    row_index: int,
    text_column: str,
    text_value: str,
    id_column: str | None,
    row_doc_type_value: str,
    row_language_value: str,
    fallback_doc_type: str,
    input_path: Path,
    model: str,
    max_chars_per_call: int,
    chunk_overlap: int,
) -> PreparedRecord:
    record_id = stable_record_id(row, id_column, row_index, text_value, text_column)
    doc_type = infer_doc_type(row_doc_type_value, fallback_doc_type, text_column)
    source_language = normalize_language(row_language_value)
    chunks = split_text(text_value, max_chars_per_call, chunk_overlap)

    chunk_rows: list[dict[str, Any]] = []
    request_lines: list[bytes] = []
    total_bytes = 0
    estimated_tokens = 0

    for chunk_index, (chunk_start, chunk_text) in enumerate(chunks):
        custom_id = custom_chunk_id(record_id, chunk_index, chunk_start)
        prompt = build_prompt(record_id, doc_type, chunk_index, chunk_start, chunk_text)
        line = build_batch_request_line(custom_id, model, prompt)
        chunk_rows.append(
            {
                "custom_id": custom_id,
                "record_id": record_id,
                "chunk_index": chunk_index,
                "chunk_start": chunk_start,
                "chunk_text": chunk_text,
            }
        )
        request_lines.append(line)
        total_bytes += len(line)
        estimated_tokens += estimate_tokens_from_request_line(line)

    metadata = {
        "record_id": record_id,
        "source_path": str(input_path),
        "source_row_index": row_index,
        "source_column": text_column,
        "source_language": source_language,
        "document_type": doc_type,
        "provider": "openai",
        "mode": "batch",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "text": text_value,
        "chunk_count": len(chunks),
    }
    return PreparedRecord(
        metadata=metadata,
        chunk_rows=chunk_rows,
        request_lines=request_lines,
        total_bytes=total_bytes,
        estimated_tokens=estimated_tokens,
    )


def prepare_batch_workdir(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir
    max_estimated_tokens_per_shard = getattr(
        args,
        "max_estimated_tokens_per_shard",
        DEFAULT_MAX_ESTIMATED_TOKENS_PER_SHARD,
    )
    manifest_file = manifest_path(work_dir)
    if work_dir.exists() and not args.overwrite:
        raise SystemExit(f"{work_dir} already exists. Pass --overwrite to replace it.")
    if work_dir.exists() and args.overwrite:
        for path in sorted(work_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if work_dir.exists():
            work_dir.rmdir()

    work_dir.mkdir(parents=True, exist_ok=True)
    requests_dir(work_dir).mkdir(parents=True, exist_ok=True)
    chunks_dir(work_dir).mkdir(parents=True, exist_ok=True)
    outputs_dir(work_dir).mkdir(parents=True, exist_ok=True)

    skip_record_ids = load_processed_ids(args.resume_output) if args.resume_output else set()
    if skip_record_ids:
        logging.info("Loaded %s processed ids from %s", len(skip_record_ids), args.resume_output)

    fields = peek_fields(args.input)
    if not fields:
        raise SystemExit(f"No rows found in {args.input}.")

    id_column = match_field(fields, args.id_column, ID_COLUMN_CANDIDATES)
    doc_type_column = match_field(fields, args.doc_type_column, DOC_TYPE_COLUMN_CANDIDATES)
    language_column = match_field(fields, args.language_column, LANGUAGE_COLUMN_CANDIDATES)
    text_specs = parse_text_specs(fields, args)

    shard_index = 0
    shard_bytes = 0
    shard_requests = 0
    shard_records = 0
    shard_estimated_tokens = 0
    shard_request_handle = None
    shard_chunk_handle = None
    shard_request_path = None
    shard_chunk_path = None
    manifest_shards: list[dict[str, Any]] = []

    def close_current_shard() -> None:
        nonlocal shard_request_handle, shard_chunk_handle, shard_request_path, shard_chunk_path
        if shard_request_handle is not None:
            shard_request_handle.close()
            shard_chunk_handle.close()
            manifest_shards.append(
                {
                    "shard_id": f"shard-{shard_index:05d}",
                    "request_path": str(shard_request_path.relative_to(work_dir)),
                    "chunk_path": str(shard_chunk_path.relative_to(work_dir)),
                    "record_count": shard_records,
                    "request_count": shard_requests,
                    "request_file_bytes": shard_bytes,
                    "estimated_tokens": shard_estimated_tokens,
                    "model": args.model,
                }
            )
            shard_request_handle = None
            shard_chunk_handle = None
            shard_request_path = None
            shard_chunk_path = None

    def open_next_shard() -> None:
        nonlocal shard_index, shard_bytes, shard_requests, shard_records, shard_estimated_tokens
        nonlocal shard_request_handle, shard_chunk_handle, shard_request_path, shard_chunk_path
        shard_index += 1
        shard_bytes = 0
        shard_requests = 0
        shard_records = 0
        shard_estimated_tokens = 0
        shard_name = f"shard-{shard_index:05d}.jsonl"
        shard_request_path = requests_dir(work_dir) / shard_name
        shard_chunk_path = chunks_dir(work_dir) / shard_name
        shard_request_handle = shard_request_path.open("w", encoding="utf-8")
        shard_chunk_handle = shard_chunk_path.open("w", encoding="utf-8")

    open_next_shard()

    prepared_records = 0
    visited_rows = 0

    with records_path(work_dir).open("w", encoding="utf-8") as records_handle:
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

                prepared = prepare_record(
                    row=row,
                    row_index=row_index,
                    text_column=text_spec.column,
                    text_value=text_value,
                    id_column=id_column,
                    row_doc_type_value=row_doc_type_value,
                    row_language_value=row_language_value,
                    fallback_doc_type=text_spec.doc_type,
                    input_path=args.input,
                    model=args.model,
                    max_chars_per_call=args.max_chars_per_call,
                    chunk_overlap=args.chunk_overlap,
                )

                if prepared.metadata["record_id"] in skip_record_ids:
                    continue

                if prepared.total_bytes > args.max_file_bytes:
                    raise SystemExit(
                        f"Single record {prepared.metadata['record_id']} needs {prepared.total_bytes} bytes, "
                        f"which exceeds --max-file-bytes={args.max_file_bytes}. Reduce chunk size."
                    )

                should_roll = (
                    shard_requests > 0
                    and (
                        shard_bytes + prepared.total_bytes > args.max_file_bytes
                        or shard_requests + len(prepared.request_lines) > args.max_requests_per_shard
                        or shard_estimated_tokens + prepared.estimated_tokens > max_estimated_tokens_per_shard
                    )
                )
                if should_roll:
                    close_current_shard()
                    open_next_shard()

                records_handle.write(json.dumps(prepared.metadata, ensure_ascii=False) + "\n")
                for line in prepared.request_lines:
                    shard_request_handle.write(line.decode("utf-8"))
                for chunk_row in prepared.chunk_rows:
                    shard_chunk_handle.write(json.dumps(chunk_row, ensure_ascii=False) + "\n")

                shard_bytes += prepared.total_bytes
                shard_requests += len(prepared.request_lines)
                shard_records += 1
                shard_estimated_tokens += prepared.estimated_tokens
                prepared_records += 1

                if prepared_records % 1000 == 0:
                    logging.info("Prepared %s documents into %s shard(s).", prepared_records, shard_index)

                if args.max_documents and prepared_records >= args.max_documents:
                    break
            if args.max_documents and prepared_records >= args.max_documents:
                break

    close_current_shard()

    manifest = {
        "provider": "openai",
        "mode": "batch",
        "batch_endpoint": DEFAULT_BATCH_ENDPOINT,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_input": str(args.input),
        "records_path": str(records_path(work_dir).relative_to(work_dir)),
        "shards": manifest_shards,
    }
    save_json(manifest_file, manifest)
    logging.info(
        "Prepared %s documents across %s request shard(s) under %s",
        prepared_records,
        len(manifest_shards),
        work_dir,
    )
    return manifest


def read_first_api_key(args: argparse.Namespace) -> str:
    api_keys = load_api_keys(args)
    if not api_keys:
        raise SystemExit("OpenAI API key missing. Pass --api-key, --api-keys-file, or set OPENAI_API_KEY / OPENAI_API_KEYS.")
    return api_keys[0]


def load_jobs_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"jobs": []}
    jobs = load_json(path)
    jobs.setdefault("jobs", [])
    return jobs


def tracked_shard_ids(jobs: dict[str, Any]) -> set[str]:
    tracked: set[str] = set()
    for job in jobs.get("jobs", []):
        shard_id = stringify(job.get("shard_id"))
        if shard_id:
            tracked.add(shard_id)
    return tracked


def rewrite_request_file_model(request_file: Path, model: str) -> int:
    updated_lines: list[str] = []
    updated_requests = 0
    with request_file.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            body = row.get("body")
            if not isinstance(body, dict):
                raise OpenAIBatchError(f"Malformed batch request in {request_file}: missing body object.")
            if body.get("model") != model:
                body["model"] = model
                updated_requests += 1
            updated_lines.append(json.dumps(row, ensure_ascii=False))
    request_file.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")
    return updated_requests


def record_ids_for_shard(chunk_file: Path) -> set[str]:
    record_ids: set[str] = set()
    with chunk_file.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            record_id = stringify(row.get("record_id"))
            if record_id:
                record_ids.add(record_id)
    return record_ids


def rewrite_records_model(records_file: Path, target_record_ids: set[str], model: str) -> int:
    temp_path = records_file.with_suffix(records_file.suffix + ".tmp")
    updated_records = 0
    with records_file.open("r", encoding="utf-8-sig") as source, temp_path.open("w", encoding="utf-8") as destination:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if stringify(row.get("record_id")) in target_record_ids:
                if row.get("model") != model:
                    row["model"] = model
                    updated_records += 1
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(records_file)
    return updated_records


def retarget_pending_shards_model(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir
    manifest = load_json(manifest_path(work_dir))
    jobs = load_jobs_if_present(jobs_path(work_dir))
    submitted_shard_ids = tracked_shard_ids(jobs)

    shards = list(manifest.get("shards", []))
    pending_shards = [shard for shard in shards if shard["shard_id"] not in submitted_shard_ids]
    if not pending_shards:
        logging.info("No unsubmitted shard found in %s. Nothing to retarget.", work_dir)
        return manifest

    shard_map = {shard["shard_id"]: shard for shard in shards}
    for job in jobs.get("jobs", []):
        shard_id = stringify(job.get("shard_id"))
        batch = job.get("batch") or {}
        batch_model = stringify(batch.get("model"))
        if shard_id and batch_model and shard_id in shard_map:
            shard_map[shard_id]["model"] = batch_model

    pending_record_ids: set[str] = set()
    updated_request_lines = 0
    for shard in pending_shards:
        request_file = work_dir / shard["request_path"]
        chunk_file = work_dir / shard["chunk_path"]
        updated_request_lines += rewrite_request_file_model(request_file, args.model)
        pending_record_ids.update(record_ids_for_shard(chunk_file))
        shard["model"] = args.model

    updated_records = rewrite_records_model(records_path(work_dir), pending_record_ids, args.model)

    manifest["model"] = args.model if not submitted_shard_ids else "mixed"
    manifest["pending_model"] = args.model
    manifest["retargeted_at_utc"] = datetime.now(timezone.utc).isoformat()
    migrations = manifest.setdefault("model_migrations", [])
    migrations.append(
        {
            "applied_at_utc": manifest["retargeted_at_utc"],
            "new_model": args.model,
            "pending_shard_count": len(pending_shards),
            "submitted_shard_count": len(submitted_shard_ids),
            "updated_request_lines": updated_request_lines,
            "updated_record_count": updated_records,
        }
    )
    save_json(manifest_path(work_dir), manifest)
    logging.info(
        "Retargeted %s pending shard(s) to %s (%s request lines, %s records). Submitted shards were left unchanged.",
        len(pending_shards),
        args.model,
        updated_request_lines,
        updated_records,
    )
    return manifest


def api_json_request(
    *,
    url: str,
    method: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    request_headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    last_error: Exception | None = None
    for attempt in range(1, DEFAULT_TRANSIENT_RETRIES + 1):
        req = request.Request(url, data=data, headers=request_headers, method=method)
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise OpenAIBatchError(f"{method} {url} returned non-object JSON.")
                return parsed
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = OpenAIBatchError(f"{method} {url} failed with HTTP {exc.code}: {body}")
            if attempt >= DEFAULT_TRANSIENT_RETRIES or not is_transient_http_status(exc.code):
                raise last_error from exc
            retry_after = parse_retry_after_seconds(exc.headers.get("Retry-After")) if exc.headers else None
            delay = retry_delay_seconds(attempt, retry_after)
            logging.warning(
                "OpenAI request failed for %s %s with HTTP %s on attempt %s/%s; retrying in %.1fs",
                method,
                url,
                exc.code,
                attempt,
                DEFAULT_TRANSIENT_RETRIES,
                delay,
            )
            time.sleep(delay)
        except error.URLError as exc:
            last_error = OpenAIBatchError(f"{method} {url} failed with network error: {exc}")
            if attempt >= DEFAULT_TRANSIENT_RETRIES:
                raise last_error from exc
            delay = retry_delay_seconds(attempt)
            logging.warning(
                "OpenAI request failed for %s %s with network error on attempt %s/%s; retrying in %.1fs",
                method,
                url,
                attempt,
                DEFAULT_TRANSIENT_RETRIES,
                delay,
            )
            time.sleep(delay)

    raise last_error if last_error else OpenAIBatchError(f"{method} {url} failed.")


def api_binary_request(
    *,
    url: str,
    method: str,
    api_key: str,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> bytes:
    request_headers = {"Authorization": f"Bearer {api_key}"}
    if headers:
        request_headers.update(headers)
    last_error: Exception | None = None
    for attempt in range(1, DEFAULT_TRANSIENT_RETRIES + 1):
        req = request.Request(url, headers=request_headers, method=method)
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                return response.read()
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = OpenAIBatchError(f"{method} {url} failed with HTTP {exc.code}: {body}")
            if attempt >= DEFAULT_TRANSIENT_RETRIES or not is_transient_http_status(exc.code):
                raise last_error from exc
            retry_after = parse_retry_after_seconds(exc.headers.get("Retry-After")) if exc.headers else None
            delay = retry_delay_seconds(attempt, retry_after)
            logging.warning(
                "OpenAI binary request failed for %s %s with HTTP %s on attempt %s/%s; retrying in %.1fs",
                method,
                url,
                exc.code,
                attempt,
                DEFAULT_TRANSIENT_RETRIES,
                delay,
            )
            time.sleep(delay)
        except error.URLError as exc:
            last_error = OpenAIBatchError(f"{method} {url} failed with network error: {exc}")
            if attempt >= DEFAULT_TRANSIENT_RETRIES:
                raise last_error from exc
            delay = retry_delay_seconds(attempt)
            logging.warning(
                "OpenAI binary request failed for %s %s with network error on attempt %s/%s; retrying in %.1fs",
                method,
                url,
                attempt,
                DEFAULT_TRANSIENT_RETRIES,
                delay,
            )
            time.sleep(delay)

    raise last_error if last_error else OpenAIBatchError(f"{method} {url} failed.")


def encode_multipart_formdata(fields: dict[str, str], file_field: str, file_path: Path, mime_type: str) -> tuple[bytes, str]:
    boundary = f"----OpenAIBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def upload_batch_input_file(api_key: str, file_path: Path, timeout_seconds: int) -> dict[str, Any]:
    body, content_type = encode_multipart_formdata({"purpose": "batch"}, "file", file_path, "application/jsonl")
    req = request.Request(
        f"{DEFAULT_ENDPOINT.rsplit('/', 2)[0]}/files",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise OpenAIBatchError(f"Uploading {file_path} failed with HTTP {exc.code}: {body_text}") from exc
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise OpenAIBatchError("File upload returned non-object JSON.")
    return payload


def submit_batches(args: argparse.Namespace) -> dict[str, Any]:
    api_key = read_first_api_key(args)
    work_dir = args.work_dir
    manifest = load_json(manifest_path(work_dir))
    jobs_file = jobs_path(work_dir)
    if jobs_file.exists() and not args.overwrite:
        jobs = load_json(jobs_file)
        jobs.setdefault("jobs", [])
    else:
        jobs = {
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "completion_window": args.completion_window,
            "jobs": [],
        }

    submitted_shard_ids = {job.get("shard_id") for job in jobs.get("jobs", [])}
    candidate_shards = list(manifest.get("shards", []))
    if args.shard_id:
        candidate_shards = [shard for shard in candidate_shards if shard["shard_id"] == args.shard_id]
        if not candidate_shards:
            raise SystemExit(f"Shard not found in manifest: {args.shard_id}")

    pending_shards = [shard for shard in candidate_shards if shard["shard_id"] not in submitted_shard_ids]
    if args.max_shards is not None and args.max_shards > 0:
        pending_shards = pending_shards[: args.max_shards]

    if not pending_shards:
        logging.info("No pending shard to submit.")
        return jobs

    for shard in pending_shards:
        request_file = work_dir / shard["request_path"]
        upload_info = upload_batch_input_file(api_key, request_file, args.timeout_seconds)
        batch_info = api_json_request(
            url=f"{DEFAULT_ENDPOINT.rsplit('/', 2)[0]}/batches",
            method="POST",
            api_key=api_key,
            payload={
                "input_file_id": upload_info["id"],
                "endpoint": DEFAULT_BATCH_ENDPOINT,
                "completion_window": args.completion_window,
                "metadata": {
                    "prompt_version": PROMPT_VERSION,
                    "source_input": str(manifest.get("source_input", ""))[:512],
                    "shard_id": shard["shard_id"],
                },
            },
            timeout_seconds=args.timeout_seconds,
        )
        jobs["jobs"].append(
            {
                "shard_id": shard["shard_id"],
                "request_path": shard["request_path"],
                "chunk_path": shard["chunk_path"],
                "request_count": shard["request_count"],
                "record_count": shard["record_count"],
                "uploaded_file_id": upload_info["id"],
                "batch": batch_info,
            }
        )
        logging.info("Submitted %s as batch %s", shard["shard_id"], batch_info["id"])

    jobs["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(jobs_file, jobs)
    logging.info("Submitted %s batch shard(s) in this run. Total tracked jobs: %s.", len(pending_shards), len(jobs["jobs"]))
    return jobs


def refresh_job_statuses(args: argparse.Namespace) -> dict[str, Any]:
    api_key = read_first_api_key(args)
    jobs = load_json(jobs_path(args.work_dir))
    base_url = DEFAULT_ENDPOINT.rsplit("/", 2)[0]

    for job in jobs.get("jobs", []):
        batch_id = job["batch"]["id"]
        batch_info = api_json_request(
            url=f"{base_url}/batches/{batch_id}",
            method="GET",
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
        )
        job["batch"] = batch_info
        logging.info(
            "%s -> %s (completed=%s failed=%s total=%s)",
            job["shard_id"],
            batch_info.get("status"),
            ((batch_info.get("request_counts") or {}).get("completed")),
            ((batch_info.get("request_counts") or {}).get("failed")),
            ((batch_info.get("request_counts") or {}).get("total")),
        )

    jobs["refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(jobs_path(args.work_dir), jobs)
    return jobs


def download_completed_outputs(args: argparse.Namespace) -> dict[str, Any]:
    api_key = read_first_api_key(args)
    jobs = load_json(jobs_path(args.work_dir))
    out_dir = outputs_dir(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = DEFAULT_ENDPOINT.rsplit("/", 2)[0]

    for job in jobs.get("jobs", []):
        batch = job["batch"]
        status = batch.get("status")
        if status != "completed":
            logging.info("Skipping %s because batch status is %s", job["shard_id"], status)
            continue

        output_file_id = batch.get("output_file_id")
        error_file_id = batch.get("error_file_id")

        if output_file_id:
            output_bytes = api_binary_request(
                url=f"{base_url}/files/{output_file_id}/content",
                method="GET",
                api_key=api_key,
                timeout_seconds=args.timeout_seconds,
            )
            local_output = out_dir / f"{job['shard_id']}.output.jsonl"
            local_output.write_bytes(output_bytes)
            job["local_output_path"] = str(local_output.relative_to(args.work_dir))
            logging.info("Downloaded output for %s -> %s", job["shard_id"], local_output)

        if error_file_id:
            error_bytes = api_binary_request(
                url=f"{base_url}/files/{error_file_id}/content",
                method="GET",
                api_key=api_key,
                timeout_seconds=args.timeout_seconds,
            )
            local_error = out_dir / f"{job['shard_id']}.error.jsonl"
            local_error.write_bytes(error_bytes)
            job["local_error_path"] = str(local_error.relative_to(args.work_dir))
            logging.info("Downloaded error file for %s -> %s", job["shard_id"], local_error)

    jobs["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(jobs_path(args.work_dir), jobs)
    return jobs


def parse_batch_output_body(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response") or {}
    status_code = response.get("status_code")
    if status_code != 200:
        raise OpenAIBatchError(f"Batch request {row.get('custom_id')} returned HTTP {status_code}")
    body = response.get("body")
    if not isinstance(body, dict):
        raise OpenAIBatchError(f"Batch request {row.get('custom_id')} returned no response body.")
    text = extract_message_text(body)
    return parse_model_json(text)


def load_chunk_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def finalize_batches(args: argparse.Namespace) -> int:
    work_dir = args.work_dir
    manifest = load_json(manifest_path(work_dir))
    jobs = load_json(jobs_path(work_dir))

    if args.output.exists() and not args.resume:
        raise SystemExit(f"{args.output} already exists. Pass --resume to append only unseen records.")

    if args.resume:
        processed_ids = load_processed_ids(args.output)
        logging.info("Loaded %s processed ids from %s", len(processed_ids), args.output)
    else:
        processed_ids = set()

    batch_outputs: dict[str, dict[str, Any]] = {}
    for job in jobs.get("jobs", []):
        local_output_rel = job.get("local_output_path")
        if not local_output_rel:
            continue
        local_output = work_dir / local_output_rel
        if not local_output.exists():
            continue
        with local_output.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if isinstance(row, dict):
                    custom_id = stringify(row.get("custom_id"))
                    if custom_id:
                        batch_outputs[custom_id] = row

    chunk_rows_by_record: dict[str, list[dict[str, Any]]] = {}
    for shard in manifest.get("shards", []):
        chunk_path = work_dir / shard["chunk_path"]
        for row in load_chunk_rows(chunk_path):
            record_id = stringify(row.get("record_id"))
            if not record_id:
                continue
            chunk_rows_by_record.setdefault(record_id, []).append(row)

    emitted_documents = 0
    error_file = error_output_path(args.output)
    records_file = work_dir / manifest["records_path"]

    with records_file.open("r", encoding="utf-8-sig") as records_handle:
        for line in records_handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            record_id = stringify(record.get("record_id"))
            if not record_id or record_id in processed_ids:
                continue

            chunk_rows = sorted(chunk_rows_by_record.get(record_id, []), key=lambda item: item["chunk_index"])
            if not chunk_rows:
                write_jsonl(error_file, {"record_id": record_id, "error": "No chunk manifest rows found for record."})
                continue

            chunks: list[tuple[int, str]] = []
            chunk_outputs: list[dict[str, Any]] = []
            missing = False
            for chunk_row in chunk_rows:
                custom_id = chunk_row["custom_id"]
                output_row = batch_outputs.get(custom_id)
                if output_row is None:
                    write_jsonl(
                        error_file,
                        {
                            "record_id": record_id,
                            "custom_id": custom_id,
                            "error": "Missing batch output row for chunk.",
                        },
                    )
                    missing = True
                    break
                try:
                    parsed_output = parse_batch_output_body(output_row)
                except Exception as exc:  # noqa: BLE001
                    write_jsonl(
                        error_file,
                        {
                            "record_id": record_id,
                            "custom_id": custom_id,
                            "error": str(exc),
                        },
                    )
                    missing = True
                    break

                chunks.append((int(chunk_row["chunk_start"]), stringify(chunk_row["chunk_text"])))
                chunk_outputs.append(parsed_output)

            if missing:
                continue

            annotation = refine_annotation(sanitize_annotation(record["text"], chunks, chunk_outputs))
            if args.require_english and annotation["document_language"] != "en":
                continue
            if args.require_it and not annotation["is_it_document"]:
                continue

            output_row = {
                "record_id": record_id,
                "source_path": record["source_path"],
                "source_row_index": record["source_row_index"],
                "source_column": record["source_column"],
                "source_language": record["source_language"],
                "document_type": record["document_type"],
                "provider": "openai",
                "mode": "batch",
                "model": record["model"],
                "prompt_version": record.get("prompt_version", PROMPT_VERSION),
                "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
                "text": record["text"],
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

    logging.info("Finalized %s document annotations into %s", emitted_documents, args.output)
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(getattr(args, "verbose", False))

    if args.command == "prepare":
        prepare_batch_workdir(args)
        return 0
    if args.command == "retarget-model":
        retarget_pending_shards_model(args)
        return 0
    if args.command == "submit":
        submit_batches(args)
        return 0
    if args.command == "status":
        refresh_job_statuses(args)
        return 0
    if args.command == "download":
        download_completed_outputs(args)
        return 0
    if args.command == "finalize":
        return finalize_batches(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
