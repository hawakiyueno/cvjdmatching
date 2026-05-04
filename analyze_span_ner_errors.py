from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from train_span_ner import (
    IGNORE_INDEX,
    SpanBasedNerModel,
    chunk_document,
    choose_device,
    load_documents,
    make_collate_fn,
    map_char_span_to_token_span,
    move_batch_to_device,
    require_transformers,
    set_seed,
    split_chunk_record,
    split_documents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze span-based NER errors for a trained checkpoint."
    )
    parser.add_argument("--annotations", required=True, type=Path, help="Annotated JSONL used for training.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to span_ner.pt checkpoint.")
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        default="test",
        help="Which split to analyze. Default: test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file for detailed analysis output.",
    )
    parser.add_argument("--device", help="Override torch device, for example cuda or cpu.")
    parser.add_argument("--max-documents", type=int, help="Optional cap before splitting.")
    parser.add_argument("--top-k", type=int, default=20, help="Max items kept in each ranked list.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


class AnalysisFeatureDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        feature = self.features[index]
        return {
            "feature_id": feature["feature_id"],
            "record_id": feature["record_id"],
            "document_type": feature["document_type"],
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"],
            "span_starts": feature["span_starts"],
            "span_ends": feature["span_ends"],
            "span_widths": feature["span_widths"],
            "labels": feature["labels"],
        }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def build_analysis_features(
    documents: list[Any],
    tokenizer: Any,
    label_to_id: dict[str, int],
    *,
    max_chars_per_example: int,
    chunk_overlap_chars: int,
    max_length: int,
    max_span_width: int,
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

    for document in documents:
        initial_chunks, unassigned_spans = chunk_document(
            document,
            max_chars_per_example=max_chars_per_example,
            chunk_overlap_chars=chunk_overlap_chars,
        )
        stats["dropped_unassigned_spans"] += unassigned_spans
        pending_chunks = list(initial_chunks)

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
            gold_char_spans: dict[tuple[int, int], dict[str, Any]] = {}
            alignment_failures = 0
            for annotation in chunk.annotations:
                token_span = map_char_span_to_token_span(offset_mapping, annotation.start, annotation.end)
                if token_span is None:
                    alignment_failures += 1
                    continue
                gold_token_spans[token_span] = label_to_id[annotation.label]
                gold_char_spans[token_span] = {
                    "label": annotation.label,
                    "start": annotation.start,
                    "end": annotation.end,
                    "text": annotation.text,
                }

            stats["dropped_alignment_spans"] += alignment_failures

            valid_tokens = [index for index, (start, end) in enumerate(offset_mapping) if end > start]
            span_starts: list[int] = []
            span_ends: list[int] = []
            span_widths: list[int] = []
            span_labels: list[int] = []
            span_char_starts: list[int] = []
            span_char_ends: list[int] = []
            span_texts: list[str] = []

            for start_position, start_token in enumerate(valid_tokens):
                max_end_position = min(start_position + max_span_width, len(valid_tokens))
                for end_position in range(start_position, max_end_position):
                    end_token = valid_tokens[end_position]
                    char_start = offset_mapping[start_token][0]
                    char_end = offset_mapping[end_token][1]
                    span_starts.append(start_token)
                    span_ends.append(end_token)
                    span_widths.append(end_position - start_position + 1)
                    span_labels.append(gold_token_spans.get((start_token, end_token), 0))
                    span_char_starts.append(char_start)
                    span_char_ends.append(char_end)
                    span_texts.append(chunk.text[char_start:char_end])

            if not span_labels:
                continue

            stats["chunk_examples"] += 1
            stats["positive_spans"] += sum(1 for label in span_labels if label != 0)
            features.append(
                {
                    "feature_id": chunk.feature_id,
                    "record_id": chunk.record_id,
                    "document_type": chunk.document_type,
                    "text": chunk.text,
                    "offset_mapping": offset_mapping,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "span_starts": span_starts,
                    "span_ends": span_ends,
                    "span_widths": span_widths,
                    "labels": span_labels,
                    "span_char_starts": span_char_starts,
                    "span_char_ends": span_char_ends,
                    "span_texts": span_texts,
                    "gold_char_spans": gold_char_spans,
                }
            )

    if not features:
        raise SystemExit("No analysis features were created.")
    return features, stats


def metric_summary(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def confidence_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p25": float(percentile(values, 0.25)),
        "p75": float(percentile(values, 0.75)),
        "min": float(min(values)),
        "max": float(max(values)),
        "count": len(values),
    }


@torch.no_grad()
def analyze(
    model: SpanBasedNerModel,
    data_loader: DataLoader,
    feature_lookup: dict[str, dict[str, Any]],
    id_to_label: dict[int, str],
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    label_stats: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0}
        for label in id_to_label.values()
        if label != "O"
    }
    false_positive_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    false_negative_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    false_positive_texts: dict[str, Counter[str]] = defaultdict(Counter)
    false_negative_texts: dict[str, Counter[str]] = defaultdict(Counter)
    confidence_tp: dict[str, list[float]] = defaultdict(list)
    confidence_fp: dict[str, list[float]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    total_loss = 0.0
    total_batches = 0

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

        probabilities = torch.softmax(outputs["logits"], dim=-1)
        predictions = outputs["logits"].argmax(dim=-1)
        labels = batch["labels"]

        for row_index, feature_id in enumerate(batch["feature_ids"]):
            feature = feature_lookup[feature_id]
            row_labels = labels[row_index].detach().cpu().tolist()
            row_predictions = predictions[row_index].detach().cpu().tolist()
            row_probabilities = probabilities[row_index].detach().cpu()

            for span_index, gold_id in enumerate(row_labels):
                if gold_id == IGNORE_INDEX:
                    continue

                predicted_id = int(row_predictions[span_index])
                if predicted_id == 0 and gold_id == 0:
                    continue

                predicted_label = id_to_label[predicted_id]
                gold_label = id_to_label[int(gold_id)]
                confidence = float(row_probabilities[span_index, predicted_id].item())
                span_text = feature["span_texts"][span_index]
                char_start = int(feature["span_char_starts"][span_index])
                char_end = int(feature["span_char_ends"][span_index])
                example = {
                    "record_id": feature["record_id"],
                    "feature_id": feature_id,
                    "document_type": feature["document_type"],
                    "text": span_text,
                    "start": char_start,
                    "end": char_end,
                    "confidence": confidence,
                }

                if predicted_id == gold_id and gold_id != 0:
                    label_stats[gold_label]["tp"] += 1
                    confidence_tp[gold_label].append(confidence)
                    continue

                if predicted_id != 0:
                    label_stats[predicted_label]["fp"] += 1
                    confidence_fp[predicted_label].append(confidence)
                    false_positive_texts[predicted_label][span_text.strip().lower()] += 1
                    if len(false_positive_examples[predicted_label]) < top_k:
                        entry = dict(example)
                        entry["gold_label"] = gold_label
                        false_positive_examples[predicted_label].append(entry)

                if gold_id != 0:
                    label_stats[gold_label]["fn"] += 1
                    false_negative_texts[gold_label][span_text.strip().lower()] += 1
                    if len(false_negative_examples[gold_label]) < top_k:
                        entry = dict(example)
                        entry["predicted_label"] = predicted_label
                        false_negative_examples[gold_label].append(entry)

                confusion[gold_label][predicted_label] += 1

    per_label = {
        label: metric_summary(stats["tp"], stats["fp"], stats["fn"])
        for label, stats in label_stats.items()
    }
    sorted_labels = sorted(
        per_label,
        key=lambda label: (per_label[label]["precision"], -per_label[label]["fp"]),
    )

    summary = {
        "loss": total_loss / max(total_batches, 1),
        "per_label": per_label,
        "sorted_by_precision": sorted_labels,
        "confidence": {
            label: {
                "tp": confidence_summary(confidence_tp[label]),
                "fp": confidence_summary(confidence_fp[label]),
            }
            for label in per_label
        },
        "top_false_positive_texts": {
            label: [
                {"text": text, "count": count}
                for text, count in counter.most_common(top_k)
                if text
            ]
            for label, counter in false_positive_texts.items()
        },
        "top_false_negative_texts": {
            label: [
                {"text": text, "count": count}
                for text, count in counter.most_common(top_k)
                if text
            ]
            for label, counter in false_negative_texts.items()
        },
        "false_positive_examples": false_positive_examples,
        "false_negative_examples": false_negative_examples,
        "confusion": {
            gold_label: dict(counter.most_common(top_k))
            for gold_label, counter in confusion.items()
        },
    }
    return summary


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    checkpoint = load_checkpoint(args.checkpoint)
    training_args = checkpoint.get("training_args") or {}
    seed = int(training_args.get("seed", 42))
    set_seed(seed)

    documents = load_documents(args.annotations, max_documents=args.max_documents)
    labeled_documents = [document for document in documents if document.annotations]
    if not labeled_documents:
        raise SystemExit("No labeled documents found for analysis.")

    label_to_id = {str(label): int(index) for label, index in (checkpoint["label_to_id"] or {}).items()}
    splits = split_documents(
        labeled_documents,
        float(training_args.get("train_ratio", 0.8)),
        float(training_args.get("dev_ratio", 0.1)),
        seed,
    )
    documents_for_split = splits[args.split]

    _, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint.parent, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    features, feature_stats = build_analysis_features(
        documents_for_split,
        tokenizer,
        label_to_id,
        max_chars_per_example=int(training_args.get("max_chars_per_example", 1400)),
        chunk_overlap_chars=int(training_args.get("chunk_overlap_chars", 120)),
        max_length=int(training_args.get("max_length", 256)),
        max_span_width=int(checkpoint["max_span_width"]),
    )

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    data_loader = DataLoader(
        AnalysisFeatureDataset(features),
        batch_size=int(training_args.get("eval_batch_size", 4)),
        shuffle=False,
        collate_fn=collate_fn,
    )

    id_to_label = {int(index): label for index, label in checkpoint["id_to_label"].items()}
    device = choose_device(args.device)
    model = SpanBasedNerModel(
        model_name=checkpoint["model_name"],
        num_labels=len(label_to_id),
        max_span_width=int(checkpoint["max_span_width"]),
        width_embedding_dim=int(checkpoint["width_embedding_dim"]),
        classifier_hidden_dim=int(checkpoint["classifier_hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    feature_lookup = {feature["feature_id"]: feature for feature in features}
    analysis = analyze(
        model=model,
        data_loader=data_loader,
        feature_lookup=feature_lookup,
        id_to_label=id_to_label,
        top_k=args.top_k,
        device=device,
    )

    output = {
        "checkpoint": str(args.checkpoint),
        "annotations": str(args.annotations),
        "split": args.split,
        "documents": len(documents_for_split),
        "features": len(features),
        "feature_stats": feature_stats,
        "analysis": analysis,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote error analysis to %s", args.output)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
