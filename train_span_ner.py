from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from djinni_gemini_ner import ENTITY_LABELS, split_text, stringify


IGNORE_INDEX = -100
NO_ENTITY_LABEL = "O"


@dataclass(frozen=True)
class SpanAnnotation:
    label: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class DocumentRecord:
    record_id: str
    document_type: str
    text: str
    annotations: tuple[SpanAnnotation, ...]


@dataclass(frozen=True)
class ChunkRecord:
    feature_id: str
    record_id: str
    document_type: str
    text: str
    annotations: tuple[SpanAnnotation, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a custom span-based NER model on Gemini-style recruitment annotations."
    )
    parser.add_argument("--annotations", required=True, type=Path, help="Input JSONL from djinni_gemini_ner.py.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for checkpoints and training artifacts.")
    parser.add_argument(
        "--model-name",
        default="roberta-base",
        help="Hugging Face encoder checkpoint. Use roberta-large on GPU for slide-aligned full training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-documents", type=int, help="Optional cap on loaded annotated documents.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Optional JSON file with fixed train/dev/test record ids. Created on first run if it does not exist.",
    )
    parser.add_argument("--max-chars-per-example", type=int, default=1400, help="Chunk long documents before tokenization.")
    parser.add_argument("--chunk-overlap-chars", type=int, default=120, help="Character overlap between training chunks.")
    parser.add_argument("--max-length", type=int, default=256, help="Max tokenizer length per example.")
    parser.add_argument("--max-span-width", type=int, default=8, help="Maximum candidate span width in tokens.")
    parser.add_argument("--negative-span-multiplier", type=int, default=4, help="Training negatives kept per positive span.")
    parser.add_argument("--min-negative-spans", type=int, default=64, help="Minimum training negatives per feature.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--resume-checkpoint", type=Path, help="Optional checkpoint to resume training from.")
    parser.add_argument(
        "--resume-additional-epochs",
        type=int,
        default=0,
        help="When resuming, train this many additional epochs beyond the checkpoint epoch.",
    )
    parser.add_argument("--train-batch-size", type=int, default=2, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Linear warmup ratio.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--width-embedding-dim", type=int, default=32, help="Span-width embedding dimension.")
    parser.add_argument("--classifier-hidden-dim", type=int, default=256, help="Hidden dimension for the span classifier head.")
    parser.add_argument("--device", help="Explicit torch device, for example cpu or cuda.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "transformers is required for train_span_ner.py. Install it with: python -m pip install transformers"
        ) from exc
    return AutoModel, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_span_annotations(row: dict[str, Any]) -> tuple[SpanAnnotation, ...]:
    text = stringify(row.get("text"))
    spans: list[SpanAnnotation] = []
    for entity in row.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        label = stringify(entity.get("label")).upper()
        if label not in ENTITY_LABELS:
            continue
        start = to_int(entity.get("start"))
        end = to_int(entity.get("end"))
        span_text = stringify(entity.get("text"))
        if start is None or end is None or start < 0 or end <= start or end > len(text):
            continue
        if text[start:end] != span_text:
            continue
        spans.append(SpanAnnotation(label=label, start=start, end=end, text=span_text))

    unique: dict[tuple[str, int, int, str], SpanAnnotation] = {}
    for span in spans:
        unique[(span.label, span.start, span.end, span.text)] = span
    return tuple(sorted(unique.values(), key=lambda item: (item.start, item.end, item.label)))


def load_documents(path: Path, max_documents: int | None = None) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            text = stringify(row.get("text"))
            record_id = stringify(row.get("record_id"))
            document_type = stringify(row.get("document_type")) or stringify(row.get("doc_type")) or "unknown"
            if not text or not record_id:
                continue
            documents.append(
                DocumentRecord(
                    record_id=record_id,
                    document_type=document_type,
                    text=text,
                    annotations=normalize_span_annotations(row),
                )
            )
            if max_documents and len(documents) >= max_documents:
                break
    return documents


def split_documents(
    documents: list[DocumentRecord],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[DocumentRecord]]:
    test_ratio = 1.0 - train_ratio - dev_ratio
    if train_ratio <= 0 or dev_ratio <= 0 or test_ratio <= 0:
        raise SystemExit("train/dev/test ratios must all be positive and sum to less than 1.0.")

    grouped: dict[str, list[DocumentRecord]] = {"cv": [], "jd": [], "other": []}
    for document in documents:
        group_key = document.document_type if document.document_type in {"cv", "jd"} else "other"
        grouped[group_key].append(document)

    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "test": []}
    for group_documents in grouped.values():
        if not group_documents:
            continue
        rng.shuffle(group_documents)
        total = len(group_documents)
        if total == 1:
            train_count, dev_count = 1, 0
        elif total == 2:
            train_count, dev_count = 1, 1
        else:
            train_count = max(1, int(total * train_ratio))
            dev_count = max(1, int(total * dev_ratio))
            if train_count + dev_count >= total:
                dev_count = 1
                train_count = max(1, total - dev_count - 1)

        splits["train"].extend(group_documents[:train_count])
        splits["dev"].extend(group_documents[train_count : train_count + dev_count])
        splits["test"].extend(group_documents[train_count + dev_count :])

    if not splits["dev"] and len(splits["train"]) > 1:
        splits["dev"].append(splits["train"].pop())
    if not splits["test"] and len(splits["train"]) > 1:
        splits["test"].append(splits["train"].pop())
    if not splits["dev"] or not splits["test"]:
        raise SystemExit("Need enough labeled documents to create non-empty dev and test splits.")

    return splits


def save_split_manifest(path: Path, splits: dict[str, list[DocumentRecord]]) -> None:
    payload = {
        split_name: [document.record_id for document in documents]
        for split_name, documents in splits.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Split manifest must be a JSON object: {path}")

    result: dict[str, set[str]] = {}
    for split_name in ("train", "dev", "test"):
        values = payload.get(split_name)
        if not isinstance(values, list) or not values:
            raise SystemExit(f"Split manifest {path} must contain a non-empty '{split_name}' list.")
        result[split_name] = {stringify(value) for value in values if stringify(value)}
        if not result[split_name]:
            raise SystemExit(f"Split manifest {path} contains no valid record ids for '{split_name}'.")
    return result


def apply_split_manifest(
    documents: list[DocumentRecord],
    manifest: dict[str, set[str]],
) -> dict[str, list[DocumentRecord]]:
    by_id = {document.record_id: document for document in documents}
    assigned_ids: set[str] = set()
    splits: dict[str, list[DocumentRecord]] = {"train": [], "dev": [], "test": []}

    for split_name, record_ids in manifest.items():
        missing = [record_id for record_id in record_ids if record_id not in by_id]
        if missing:
            raise SystemExit(
                f"Split manifest contains {len(missing)} record ids missing from current annotations, "
                f"for example: {missing[:5]}"
            )
        for record_id in sorted(record_ids):
            splits[split_name].append(by_id[record_id])
            assigned_ids.add(record_id)

    unassigned = [document.record_id for document in documents if document.record_id not in assigned_ids]
    if unassigned:
        raise SystemExit(
            f"Split manifest does not cover {len(unassigned)} annotated documents, "
            f"for example: {unassigned[:5]}"
        )

    return splits


def chunk_document(
    document: DocumentRecord,
    max_chars_per_example: int,
    chunk_overlap_chars: int,
) -> tuple[list[ChunkRecord], int]:
    chunks = split_text(document.text, max_chars_per_example, chunk_overlap_chars)
    remaining_annotations = list(document.annotations)
    chunk_records: list[ChunkRecord] = []

    for chunk_index, (chunk_start, chunk_text) in enumerate(chunks):
        chunk_end = chunk_start + len(chunk_text)
        chunk_annotations: list[SpanAnnotation] = []
        next_remaining: list[SpanAnnotation] = []

        for annotation in remaining_annotations:
            if annotation.start >= chunk_start and annotation.end <= chunk_end:
                chunk_annotations.append(
                    SpanAnnotation(
                        label=annotation.label,
                        start=annotation.start - chunk_start,
                        end=annotation.end - chunk_start,
                        text=annotation.text,
                    )
                )
            else:
                next_remaining.append(annotation)

        remaining_annotations = next_remaining
        chunk_records.append(
            ChunkRecord(
                feature_id=f"{document.record_id}::chunk-{chunk_index}",
                record_id=document.record_id,
                document_type=document.document_type,
                text=chunk_text,
                annotations=tuple(chunk_annotations),
            )
        )

    return chunk_records, len(remaining_annotations)


def split_chunk_record(
    chunk: ChunkRecord,
    max_chars_per_example: int,
    chunk_overlap_chars: int,
) -> tuple[list[ChunkRecord], int]:
    subchunks = split_text(chunk.text, max_chars_per_example, chunk_overlap_chars)
    remaining_annotations = list(chunk.annotations)
    chunk_records: list[ChunkRecord] = []

    for chunk_index, (chunk_start, chunk_text) in enumerate(subchunks):
        chunk_end = chunk_start + len(chunk_text)
        chunk_annotations: list[SpanAnnotation] = []
        next_remaining: list[SpanAnnotation] = []

        for annotation in remaining_annotations:
            if annotation.start >= chunk_start and annotation.end <= chunk_end:
                chunk_annotations.append(
                    SpanAnnotation(
                        label=annotation.label,
                        start=annotation.start - chunk_start,
                        end=annotation.end - chunk_start,
                        text=annotation.text,
                    )
                )
            else:
                next_remaining.append(annotation)

        remaining_annotations = next_remaining
        chunk_records.append(
            ChunkRecord(
                feature_id=f"{chunk.feature_id}::sub-{chunk_index}",
                record_id=chunk.record_id,
                document_type=chunk.document_type,
                text=chunk_text,
                annotations=tuple(chunk_annotations),
            )
        )

    return chunk_records, len(remaining_annotations)


def build_label_vocabulary(documents: list[DocumentRecord]) -> list[str]:
    labels = sorted({annotation.label for document in documents for annotation in document.annotations})
    if not labels:
        raise SystemExit("No valid entity spans found in the annotations file.")
    return [NO_ENTITY_LABEL] + labels


def map_char_span_to_token_span(
    offset_mapping: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    token_start = None
    token_end = None
    for index, (start, end) in enumerate(offset_mapping):
        if end <= start:
            continue
        if start == char_start:
            token_start = index
        if end == char_end:
            token_end = index

    if token_start is not None and token_end is not None and token_start <= token_end:
        return token_start, token_end

    overlapping = [
        index
        for index, (start, end) in enumerate(offset_mapping)
        if end > start and not (end <= char_start or start >= char_end)
    ]
    if not overlapping:
        return None
    if offset_mapping[overlapping[0]][0] != char_start or offset_mapping[overlapping[-1]][1] != char_end:
        return None
    return overlapping[0], overlapping[-1]


def enumerate_candidate_spans(
    offset_mapping: list[tuple[int, int]],
    gold_token_spans: dict[tuple[int, int], int],
    max_span_width: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    valid_tokens = [index for index, (start, end) in enumerate(offset_mapping) if end > start]
    span_starts: list[int] = []
    span_ends: list[int] = []
    span_widths: list[int] = []
    span_labels: list[int] = []

    for start_position, start_token in enumerate(valid_tokens):
        max_end_position = min(start_position + max_span_width, len(valid_tokens))
        for end_position in range(start_position, max_end_position):
            end_token = valid_tokens[end_position]
            span_starts.append(start_token)
            span_ends.append(end_token)
            span_widths.append(end_position - start_position + 1)
            span_labels.append(gold_token_spans.get((start_token, end_token), 0))

    return span_starts, span_ends, span_widths, span_labels


def sample_training_spans(
    span_starts: list[int],
    span_ends: list[int],
    span_widths: list[int],
    span_labels: list[int],
    negative_span_multiplier: int,
    min_negative_spans: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], list[int]]:
    positive_indices = [index for index, label in enumerate(span_labels) if label != 0]
    negative_indices = [index for index, label in enumerate(span_labels) if label == 0]
    if not negative_indices:
        return span_starts, span_ends, span_widths, span_labels

    if positive_indices:
        keep_negative_count = max(min_negative_spans, len(positive_indices) * negative_span_multiplier)
    else:
        keep_negative_count = min_negative_spans
    keep_negative_count = min(keep_negative_count, len(negative_indices))
    kept_indices = sorted(positive_indices + rng.sample(negative_indices, keep_negative_count))
    return (
        [span_starts[index] for index in kept_indices],
        [span_ends[index] for index in kept_indices],
        [span_widths[index] for index in kept_indices],
        [span_labels[index] for index in kept_indices],
    )


def build_features(
    documents: list[DocumentRecord],
    tokenizer: Any,
    label_to_id: dict[str, int],
    *,
    split_name: str,
    max_chars_per_example: int,
    chunk_overlap_chars: int,
    max_length: int,
    max_span_width: int,
    negative_span_multiplier: int,
    min_negative_spans: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features: list[dict[str, Any]] = []
    stats = {
        "documents": len(documents),
        "chunk_examples": 0,
        "dropped_unassigned_spans": 0,
        "dropped_truncated_chunks": 0,
        "retokenized_subchunks": 0,
        "dropped_alignment_spans": 0,
        "positive_spans": 0,
    }

    for document_index, document in enumerate(documents):
        initial_chunk_records, unassigned_spans = chunk_document(
            document,
            max_chars_per_example=max_chars_per_example,
            chunk_overlap_chars=chunk_overlap_chars,
        )
        stats["dropped_unassigned_spans"] += unassigned_spans
        pending_chunks = list(initial_chunk_records)

        chunk_index = 0
        while pending_chunks:
            chunk = pending_chunks.pop(0)
            tokenized = tokenizer(
                chunk.text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            offset_mapping = [tuple(item) for item in tokenized["offset_mapping"]]

            if max(end for _, end in offset_mapping) < len(chunk.text):
                smaller_max_chars = max(min(len(chunk.text) // 2, max_chars_per_example // 2), 160)
                if len(chunk.text) > smaller_max_chars + 20:
                    subchunks, extra_unassigned = split_chunk_record(
                        chunk,
                        max_chars_per_example=smaller_max_chars,
                        chunk_overlap_chars=min(chunk_overlap_chars, max(40, smaller_max_chars // 8)),
                    )
                    stats["retokenized_subchunks"] += len(subchunks)
                    stats["dropped_unassigned_spans"] += extra_unassigned
                    pending_chunks = subchunks + pending_chunks
                else:
                    stats["dropped_truncated_chunks"] += 1
                continue

            gold_token_spans: dict[tuple[int, int], int] = {}
            alignment_failures = 0
            for annotation in chunk.annotations:
                token_span = map_char_span_to_token_span(offset_mapping, annotation.start, annotation.end)
                if token_span is None:
                    alignment_failures += 1
                    continue
                gold_token_spans[token_span] = label_to_id[annotation.label]

            stats["dropped_alignment_spans"] += alignment_failures
            span_starts, span_ends, span_widths, span_labels = enumerate_candidate_spans(
                offset_mapping=offset_mapping,
                gold_token_spans=gold_token_spans,
                max_span_width=max_span_width,
            )

            if split_name == "train":
                rng = random.Random(f"{seed}:{document_index}:{chunk_index}")
                span_starts, span_ends, span_widths, span_labels = sample_training_spans(
                    span_starts=span_starts,
                    span_ends=span_ends,
                    span_widths=span_widths,
                    span_labels=span_labels,
                    negative_span_multiplier=negative_span_multiplier,
                    min_negative_spans=min_negative_spans,
                    rng=rng,
                )

            if not span_labels:
                continue

            stats["chunk_examples"] += 1
            stats["positive_spans"] += sum(1 for label in span_labels if label != 0)
            features.append(
                {
                    "feature_id": chunk.feature_id,
                    "record_id": chunk.record_id,
                    "document_type": chunk.document_type,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "span_starts": span_starts,
                    "span_ends": span_ends,
                    "span_widths": span_widths,
                    "labels": span_labels,
                }
            )
            chunk_index += 1

    if not features:
        raise SystemExit(f"No usable {split_name} features were created. Check the annotations and max lengths.")
    return features, stats


class SpanFeatureDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
        batch_size = len(batch)
        max_seq_len = max(len(item["input_ids"]) for item in batch)
        max_spans = max(len(item["labels"]) for item in batch)

        input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        span_starts = torch.zeros((batch_size, max_spans), dtype=torch.long)
        span_ends = torch.zeros((batch_size, max_spans), dtype=torch.long)
        span_widths = torch.zeros((batch_size, max_spans), dtype=torch.long)
        labels = torch.full((batch_size, max_spans), IGNORE_INDEX, dtype=torch.long)
        feature_ids: list[str] = []

        for row_index, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            span_count = len(item["labels"])
            input_ids[row_index, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[row_index, :seq_len] = torch.tensor(item["attention_mask"], dtype=torch.long)
            span_starts[row_index, :span_count] = torch.tensor(item["span_starts"], dtype=torch.long)
            span_ends[row_index, :span_count] = torch.tensor(item["span_ends"], dtype=torch.long)
            span_widths[row_index, :span_count] = torch.tensor(item["span_widths"], dtype=torch.long)
            labels[row_index, :span_count] = torch.tensor(item["labels"], dtype=torch.long)
            feature_ids.append(str(item["feature_id"]))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_starts": span_starts,
            "span_ends": span_ends,
            "span_widths": span_widths,
            "labels": labels,
            "feature_ids": feature_ids,
        }

    return collate_fn


class SpanBasedNerModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        max_span_width: int,
        width_embedding_dim: int,
        classifier_hidden_dim: int,
    ) -> None:
        super().__init__()
        AutoModel, _ = require_transformers()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.width_embeddings = nn.Embedding(max_span_width + 1, width_embedding_dim)
        self.dropout = nn.Dropout(float(getattr(self.encoder.config, "hidden_dropout_prob", 0.1)))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3 + width_embedding_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
        span_widths: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        batch_size = sequence_output.size(0)
        batch_indices = torch.arange(batch_size, device=sequence_output.device).unsqueeze(1)
        start_repr = sequence_output[batch_indices, span_starts]
        end_repr = sequence_output[batch_indices, span_ends]
        width_repr = self.width_embeddings(span_widths)
        span_repr = torch.cat([start_repr, end_repr, start_repr * end_repr, width_repr], dim=-1)
        logits = self.classifier(self.dropout(span_repr))

        result = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)
            result["loss"] = loss
        return result


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        remaining_steps = max(total_steps - current_step, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        return max(float(remaining_steps) / float(decay_steps), 0.0)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(model: SpanBasedNerModel, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            span_starts=batch["span_starts"],
            span_ends=batch["span_ends"],
            span_widths=batch["span_widths"],
            labels=batch["labels"],
        )
        total_loss += float(outputs["loss"].item())
        total_batches += 1

        predictions = outputs["logits"].argmax(dim=-1)
        labels = batch["labels"]
        valid_mask = labels != IGNORE_INDEX
        valid_predictions = predictions[valid_mask]
        valid_labels = labels[valid_mask]

        for predicted, gold in zip(valid_predictions.tolist(), valid_labels.tolist(), strict=False):
            if predicted == gold and gold != 0:
                true_positive += 1
            else:
                if predicted != 0:
                    false_positive += 1
                if gold != 0:
                    false_negative += 1

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": total_loss / max(total_batches, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_checkpoint(
    output_dir: Path,
    *,
    model: SpanBasedNerModel,
    tokenizer: Any,
    label_to_id: dict[str, int],
    optimizer: AdamW | None,
    scheduler: LambdaLR | None,
    args: argparse.Namespace,
    completed_epoch: int,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "label_to_id": label_to_id,
            "id_to_label": {index: label for label, index in label_to_id.items()},
            "model_name": args.model_name,
            "max_span_width": args.max_span_width,
            "width_embedding_dim": args.width_embedding_dim,
            "classifier_hidden_dim": args.classifier_hidden_dim,
            "completed_epoch": completed_epoch,
            "metrics": metrics,
            "training_args": serializable_args,
        },
        output_dir / "span_ner.pt",
    )


def train_epoch(
    model: SpanBasedNerModel,
    data_loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_batches = 0

    for step, batch in enumerate(data_loader, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            span_starts=batch["span_starts"],
            span_ends=batch["span_ends"],
            span_widths=batch["span_widths"],
            labels=batch["labels"],
        )
        loss = outputs["loss"] / gradient_accumulation_steps
        loss.backward()

        total_loss += float(outputs["loss"].item())
        total_batches += 1

        if step % gradient_accumulation_steps == 0 or step == len(data_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_batches, 1)


def choose_device(explicit_device: str | None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_completed_epoch(checkpoint: dict[str, Any]) -> int:
    explicit_epoch = checkpoint.get("completed_epoch")
    if explicit_epoch is not None:
        return int(explicit_epoch)
    metrics = checkpoint.get("metrics")
    if isinstance(metrics, dict) and metrics.get("epoch") is not None:
        return int(metrics["epoch"])
    return 0


def resolve_epoch_plan(requested_epochs: int, resume_epoch: int, resume_additional_epochs: int) -> tuple[int, int]:
    if resume_epoch < 0:
        raise SystemExit("Resume checkpoint epoch must be non-negative.")
    if resume_additional_epochs < 0:
        raise SystemExit("--resume-additional-epochs must be non-negative.")

    if resume_epoch == 0:
        return 1, requested_epochs

    start_epoch = resume_epoch + 1
    if resume_additional_epochs > 0:
        target_total_epochs = resume_epoch + resume_additional_epochs
    else:
        target_total_epochs = requested_epochs

    if target_total_epochs < start_epoch:
        raise SystemExit(
            f"Nothing to do: resume checkpoint is already at epoch {resume_epoch}, "
            f"but target total epochs is {target_total_epochs}."
        )
    return start_epoch, target_total_epochs


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    label_to_id: dict[str, int],
    args: argparse.Namespace,
) -> None:
    checkpoint_labels = checkpoint.get("label_to_id")
    if checkpoint_labels and checkpoint_labels != label_to_id:
        raise SystemExit("Resume checkpoint label set does not match the current annotations.")

    for key in ("max_span_width", "width_embedding_dim", "classifier_hidden_dim", "model_name"):
        checkpoint_value = checkpoint.get(key)
        current_value = getattr(args, key)
        if checkpoint_value is not None and checkpoint_value != current_value:
            raise SystemExit(
                f"Resume checkpoint {key}={checkpoint_value!r} does not match current arg value {current_value!r}."
            )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    set_seed(args.seed)

    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if args.resume_additional_epochs < 0:
        raise SystemExit("--resume-additional-epochs must be non-negative.")
    if not args.annotations.exists():
        raise SystemExit(f"Annotations file not found: {args.annotations}")
    if args.resume_checkpoint and not args.resume_checkpoint.exists():
        raise SystemExit(f"Resume checkpoint not found: {args.resume_checkpoint}")

    AutoModel, AutoTokenizer = require_transformers()
    del AutoModel

    documents = load_documents(args.annotations, max_documents=args.max_documents)
    if not documents:
        raise SystemExit(f"No annotated documents found in {args.annotations}")

    labeled_documents = [document for document in documents if document.annotations]
    if not labeled_documents:
        raise SystemExit(f"No entity spans found in {args.annotations}")
    if len(labeled_documents) < 3:
        raise SystemExit("At least 3 labeled documents are required for train/dev/test splitting.")

    label_list = build_label_vocabulary(labeled_documents)
    label_to_id = {label: index for index, label in enumerate(label_list)}
    if args.split_manifest:
        if args.split_manifest.exists():
            manifest = load_split_manifest(args.split_manifest)
            splits = apply_split_manifest(labeled_documents, manifest)
            logging.info("Loaded fixed split manifest from %s.", args.split_manifest)
        else:
            splits = split_documents(labeled_documents, args.train_ratio, args.dev_ratio, args.seed)
            save_split_manifest(args.split_manifest, splits)
            logging.info("Created fixed split manifest at %s.", args.split_manifest)
    else:
        splits = split_documents(labeled_documents, args.train_ratio, args.dev_ratio, args.seed)
    logging.info(
        "Loaded %s labeled documents -> train=%s dev=%s test=%s",
        len(labeled_documents),
        len(splits["train"]),
        len(splits["dev"]),
        len(splits["test"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_features, train_stats = build_features(
        splits["train"],
        tokenizer,
        label_to_id,
        split_name="train",
        max_chars_per_example=args.max_chars_per_example,
        chunk_overlap_chars=args.chunk_overlap_chars,
        max_length=args.max_length,
        max_span_width=args.max_span_width,
        negative_span_multiplier=args.negative_span_multiplier,
        min_negative_spans=args.min_negative_spans,
        seed=args.seed,
    )
    dev_features, dev_stats = build_features(
        splits["dev"],
        tokenizer,
        label_to_id,
        split_name="dev",
        max_chars_per_example=args.max_chars_per_example,
        chunk_overlap_chars=args.chunk_overlap_chars,
        max_length=args.max_length,
        max_span_width=args.max_span_width,
        negative_span_multiplier=args.negative_span_multiplier,
        min_negative_spans=args.min_negative_spans,
        seed=args.seed,
    )
    test_features, test_stats = build_features(
        splits["test"],
        tokenizer,
        label_to_id,
        split_name="test",
        max_chars_per_example=args.max_chars_per_example,
        chunk_overlap_chars=args.chunk_overlap_chars,
        max_length=args.max_length,
        max_span_width=args.max_span_width,
        negative_span_multiplier=args.negative_span_multiplier,
        min_negative_spans=args.min_negative_spans,
        seed=args.seed,
    )
    logging.info(
        "Prepared features -> train=%s dev=%s test=%s",
        len(train_features),
        len(dev_features),
        len(test_features),
    )

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        SpanFeatureDataset(train_features),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        SpanFeatureDataset(dev_features),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        SpanFeatureDataset(test_features),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = choose_device(args.device)
    model = SpanBasedNerModel(
        model_name=args.model_name,
        num_labels=len(label_list),
        max_span_width=args.max_span_width,
        width_embedding_dim=args.width_embedding_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = max(math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1)), 1)
    resume_checkpoint: dict[str, Any] | None = None
    resumed_epoch = 0
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        validate_resume_checkpoint(resume_checkpoint, label_to_id=label_to_id, args=args)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        resumed_epoch = checkpoint_completed_epoch(resume_checkpoint)
    start_epoch, target_total_epochs = resolve_epoch_plan(args.epochs, resumed_epoch, args.resume_additional_epochs)
    total_steps = steps_per_epoch * target_total_epochs
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio)

    best_metrics: dict[str, Any] | None = None
    best_dev_f1 = -1.0
    if resume_checkpoint:
        checkpoint_metrics = resume_checkpoint.get("metrics")
        if isinstance(checkpoint_metrics, dict):
            best_metrics = checkpoint_metrics
            best_dev_f1 = float((checkpoint_metrics.get("dev") or {}).get("f1", -1.0))

        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        scheduler_state = resume_checkpoint.get("scheduler_state_dict")
        if optimizer_state and scheduler_state:
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            logging.info(
                "Resumed optimizer and scheduler state from %s at epoch %s.",
                args.resume_checkpoint,
                resumed_epoch,
            )
        else:
            logging.warning(
                "Resume checkpoint %s does not contain optimizer/scheduler state. "
                "Continuing from model weights only at epoch %s.",
                args.resume_checkpoint,
                resumed_epoch,
            )

    for epoch in range(start_epoch, target_total_epochs + 1):
        train_loss = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=max(args.gradient_accumulation_steps, 1),
            max_grad_norm=args.max_grad_norm,
        )
        dev_metrics = evaluate(model, dev_loader, device)
        logging.info(
            "Epoch %s/%s | train_loss=%.4f | dev_loss=%.4f | dev_f1=%.4f",
            epoch,
            target_total_epochs,
            train_loss,
            dev_metrics["loss"],
            dev_metrics["f1"],
        )
        if dev_metrics["f1"] >= best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev": dev_metrics,
            }
            save_checkpoint(
                args.output_dir,
                model=model,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                completed_epoch=epoch,
                metrics=best_metrics,
            )

    assert best_metrics is not None
    best_checkpoint_path = args.output_dir / "span_ner.pt"
    if not best_checkpoint_path.exists():
        if args.resume_checkpoint is None:
            raise SystemExit(f"Best checkpoint not found: {best_checkpoint_path}")
        best_checkpoint_path = args.resume_checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    summary = {
        "best": best_metrics,
        "test": test_metrics,
        "labels": label_list,
        "splits": {name: len(items) for name, items in splits.items()},
        "feature_stats": {
            "train": train_stats,
            "dev": dev_stats,
            "test": test_stats,
        },
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "device": str(device),
        "model_name": args.model_name,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logging.info(
        "Finished training. Best dev_f1=%.4f | test_f1=%.4f | checkpoint=%s",
        best_metrics["dev"]["f1"],
        test_metrics["f1"],
        best_checkpoint_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
