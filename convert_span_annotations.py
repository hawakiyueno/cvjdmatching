from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from djinni_gemini_ner import ENTITY_LABELS, normalize_doc_type, stringify


SOURCE_LABEL_ALIASES = {
    "SKILL": "SKILL",
    "SKILLS": "SKILL",
    "HARD_SKILL": "SKILL",
    "SOFT_SKILL": "SKILL",
    "OCCUPATION": "JOB_ROLE",
    "ROLE": "JOB_ROLE",
    "JOB_ROLE": "JOB_ROLE",
    "QUALIFICATION": "DEGREE",
    "EDUCATION": "DEGREE",
    "DEGREE": "DEGREE",
    "CERTIFICATION": "CERTIFICATION",
    "DOMAIN": "INDUSTRY",
    "EXPERIENCE": "WORK_ACTIVITY",
    "RESPONSIBILITY": "WORK_ACTIVITY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Doccano-style or generic span annotations into the training JSONL expected by train_span_ner.py."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL or JSON file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL for train_span_ner.py.")
    parser.add_argument("--text-field", default="text", help="Field containing the document text.")
    parser.add_argument("--id-field", default="doc_id", help="Field containing a stable document id.")
    parser.add_argument("--doc-type-field", help="Optional field containing cv/jd metadata.")
    parser.add_argument("--document-type", default="unknown", help="Fallback document type when no field is provided.")
    parser.add_argument(
        "--label-map-json",
        help="Optional JSON object mapping source labels to target labels, for example '{\"Skills\":\"TECHNOLOGY\"}'.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    return parser.parse_args()


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} already exists. Pass --overwrite to replace it.")


def ensure_readable(path: Path) -> None:
    if path.exists():
        return
    siblings = sorted(candidate.name for candidate in path.parent.glob("*.jsonl"))
    hint = ""
    if siblings:
        hint = "\nAvailable .jsonl files in this folder:\n- " + "\n- ".join(siblings[:20])
    raise SystemExit(f"Input file not found: {path}{hint}")


def parse_label_map(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("--label-map-json must be a JSON object.")
    return {str(key): str(value) for key, value in payload.items()}


def iter_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("data", "items", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
        raise SystemExit(f"Unsupported JSON structure in {path}")

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


def normalize_label(label: str, label_map: dict[str, str]) -> str | None:
    raw = label.strip()
    prefix = raw.split(":", 1)[0].strip()
    mapped = label_map.get(raw, label_map.get(prefix, label_map.get(raw.upper(), label_map.get(prefix.upper(), prefix))))
    candidate = mapped.strip().upper().replace(" ", "_")
    candidate = SOURCE_LABEL_ALIASES.get(candidate, candidate)
    return candidate if candidate in ENTITY_LABELS else None


def parse_span_triplet(item: Any) -> tuple[int, int, str] | None:
    if not isinstance(item, list) or len(item) != 3:
        return None
    try:
        start = int(item[0])
        end = int(item[1])
    except (TypeError, ValueError):
        return None
    label = stringify(item[2])
    if not label:
        return None
    return start, end, label


def parse_span_object(item: Any) -> tuple[int, int, str] | None:
    if not isinstance(item, dict):
        return None
    label = stringify(item.get("label") or item.get("entity") or item.get("tag"))
    if not label:
        return None
    try:
        start = int(item.get("start"))
        end = int(item.get("end"))
    except (TypeError, ValueError):
        return None
    return start, end, label


def extract_entities(text: str, row: dict[str, Any], label_map: dict[str, str]) -> list[dict[str, Any]]:
    raw_labels = row.get("labels")
    raw_annotations = row.get("annotations")
    spans: list[dict[str, Any]] = []

    if isinstance(raw_labels, list):
        for item in raw_labels:
            parsed = parse_span_triplet(item)
            if not parsed:
                continue
            start, end, label = parsed
            normalized_label = normalize_label(label, label_map)
            if normalized_label is None or start < 0 or end <= start or end > len(text):
                continue
            spans.append(
                {
                    "label": normalized_label,
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "normalized": text[start:end].lower(),
                }
            )

    if isinstance(raw_annotations, list):
        for item in raw_annotations:
            parsed = parse_span_object(item)
            if not parsed:
                continue
            start, end, label = parsed
            normalized_label = normalize_label(label, label_map)
            if normalized_label is None or start < 0 or end <= start or end > len(text):
                continue
            spans.append(
                {
                    "label": normalized_label,
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "normalized": text[start:end].lower(),
                }
            )

    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for span in spans:
        unique[(span["label"], span["start"], span["end"])] = span
    return sorted(unique.values(), key=lambda item: (item["start"], item["end"], item["label"]))


def build_output_row(
    row: dict[str, Any],
    index: int,
    text_field: str,
    id_field: str,
    doc_type_field: str | None,
    default_doc_type: str,
    label_map: dict[str, str],
) -> dict[str, Any] | None:
    text = stringify(row.get(text_field))
    if not text:
        return None

    raw_id = stringify(row.get(id_field)) or f"row-{index}"
    raw_doc_type = stringify(row.get(doc_type_field)) if doc_type_field else ""
    document_type = normalize_doc_type(raw_doc_type) if raw_doc_type else normalize_doc_type(default_doc_type)
    if document_type == "unknown":
        document_type = default_doc_type

    entities = extract_entities(text, row, label_map)
    return {
        "record_id": str(raw_id),
        "document_type": document_type,
        "text": text,
        "entities": entities,
        "qualification_facts": [],
    }


def main() -> int:
    args = parse_args()
    ensure_writable(args.output, args.overwrite)
    ensure_readable(args.input)
    label_map = parse_label_map(args.label_map_json)
    rows = iter_rows(args.input)

    converted = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            output_row = build_output_row(
                row=row,
                index=index,
                text_field=args.text_field,
                id_field=args.id_field,
                doc_type_field=args.doc_type_field,
                default_doc_type=args.document_type,
                label_map=label_map,
            )
            if output_row is None:
                continue
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted {converted} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
