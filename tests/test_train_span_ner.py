import unittest

from train_span_ner import (
    DocumentRecord,
    SpanAnnotation,
    apply_split_manifest,
    chunk_document,
    checkpoint_completed_epoch,
    load_split_manifest,
    map_char_span_to_token_span,
    normalize_span_annotations,
    resolve_epoch_plan,
    sample_training_spans,
    save_split_manifest,
    split_chunk_record,
    split_documents,
)


class TrainSpanNerTests(unittest.TestCase):
    def test_normalize_span_annotations_keeps_valid_entity(self) -> None:
        row = {
            "text": "Python developer with AWS experience",
            "entities": [
                {"label": "TECHNOLOGY", "start": 0, "end": 6, "text": "Python"},
                {"label": "UNKNOWN", "start": 0, "end": 6, "text": "Python"},
                {"label": "JOB_ROLE", "start": 7, "end": 16, "text": "developerx"},
            ],
        }
        spans = normalize_span_annotations(row)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].label, "TECHNOLOGY")

    def test_chunk_document_assigns_overlap_span_once(self) -> None:
        text = "0123456789abcdefghij"
        document = DocumentRecord(
            record_id="doc-1",
            document_type="cv",
            text=text,
            annotations=(
                SpanAnnotation(label="TECHNOLOGY", start=7, end=9, text=text[7:9]),
                SpanAnnotation(label="JOB_ROLE", start=12, end=15, text=text[12:15]),
            ),
        )
        chunks, unassigned = chunk_document(document, max_chars_per_example=10, chunk_overlap_chars=3)
        counts = sum(len(chunk.annotations) for chunk in chunks)
        first_chunk_texts = [annotation.text for annotation in chunks[0].annotations]
        second_chunk_texts = [annotation.text for annotation in chunks[1].annotations]

        self.assertEqual(unassigned, 0)
        self.assertEqual(counts, 2)
        self.assertIn("78", first_chunk_texts)
        self.assertNotIn("78", second_chunk_texts)

    def test_map_char_span_to_token_span_prefers_exact_boundaries(self) -> None:
        offsets = [(0, 0), (0, 4), (5, 11), (12, 15), (0, 0)]
        self.assertEqual(map_char_span_to_token_span(offsets, 5, 11), (2, 2))
        self.assertEqual(map_char_span_to_token_span(offsets, 0, 11), (1, 2))
        self.assertIsNone(map_char_span_to_token_span(offsets, 1, 11))

    def test_split_chunk_record_preserves_relative_annotations(self) -> None:
        from train_span_ner import ChunkRecord

        chunk = ChunkRecord(
            feature_id="doc-1::chunk-0",
            record_id="doc-1",
            document_type="cv",
            text="abcdefghij1234567890",
            annotations=(
                SpanAnnotation(label="TECHNOLOGY", start=2, end=5, text="cde"),
                SpanAnnotation(label="JOB_ROLE", start=12, end=16, text="3456"),
            ),
        )
        subchunks, unassigned = split_chunk_record(chunk, max_chars_per_example=10, chunk_overlap_chars=2)
        flattened = [(annotation.label, annotation.text) for subchunk in subchunks for annotation in subchunk.annotations]

        self.assertEqual(unassigned, 0)
        self.assertEqual(flattened, [("TECHNOLOGY", "cde"), ("JOB_ROLE", "3456")])

    def test_sample_training_spans_keeps_all_positive_labels(self) -> None:
        starts, ends, widths, labels = sample_training_spans(
            span_starts=[1, 2, 3, 4, 5],
            span_ends=[1, 2, 3, 4, 5],
            span_widths=[1, 1, 1, 1, 1],
            span_labels=[0, 2, 0, 3, 0],
            negative_span_multiplier=1,
            min_negative_spans=1,
            rng=__import__("random").Random(7),
        )
        self.assertIn(2, labels)
        self.assertIn(3, labels)
        self.assertEqual(len(starts), len(labels))
        self.assertEqual(len(ends), len(labels))
        self.assertEqual(len(widths), len(labels))

    def test_split_documents_backfills_dev_and_test(self) -> None:
        documents = [
            DocumentRecord("a", "cv", "t", (SpanAnnotation("TECHNOLOGY", 0, 1, "t"),)),
            DocumentRecord("b", "cv", "t", (SpanAnnotation("TECHNOLOGY", 0, 1, "t"),)),
            DocumentRecord("c", "cv", "t", (SpanAnnotation("TECHNOLOGY", 0, 1, "t"),)),
        ]
        splits = split_documents(documents, train_ratio=0.8, dev_ratio=0.1, seed=7)
        self.assertTrue(splits["train"])
        self.assertTrue(splits["dev"])
        self.assertTrue(splits["test"])

    def test_checkpoint_completed_epoch_prefers_explicit_field(self) -> None:
        checkpoint = {"completed_epoch": 6, "metrics": {"epoch": 5}}
        self.assertEqual(checkpoint_completed_epoch(checkpoint), 6)

    def test_checkpoint_completed_epoch_falls_back_to_metrics(self) -> None:
        checkpoint = {"metrics": {"epoch": 4}}
        self.assertEqual(checkpoint_completed_epoch(checkpoint), 4)

    def test_resolve_epoch_plan_for_new_training(self) -> None:
        self.assertEqual(resolve_epoch_plan(6, 0, 0), (1, 6))

    def test_resolve_epoch_plan_for_interrupted_resume(self) -> None:
        self.assertEqual(resolve_epoch_plan(9, 6, 0), (7, 9))

    def test_resolve_epoch_plan_for_additional_epochs(self) -> None:
        self.assertEqual(resolve_epoch_plan(6, 6, 3), (7, 9))

    def test_split_manifest_round_trip(self) -> None:
        import tempfile
        from pathlib import Path

        documents = [
            DocumentRecord("a", "cv", "t", (SpanAnnotation("TECHNOLOGY", 0, 1, "t"),)),
            DocumentRecord("b", "cv", "t", (SpanAnnotation("TECHNOLOGY", 0, 1, "t"),)),
            DocumentRecord("c", "jd", "t", (SpanAnnotation("TECHNOLOGY", 0, 1, "t"),)),
        ]
        splits = {"train": [documents[0]], "dev": [documents[1]], "test": [documents[2]]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "splits.json"
            save_split_manifest(path, splits)
            manifest = load_split_manifest(path)
            applied = apply_split_manifest(documents, manifest)

        self.assertEqual([document.record_id for document in applied["train"]], ["a"])
        self.assertEqual([document.record_id for document in applied["dev"]], ["b"])
        self.assertEqual([document.record_id for document in applied["test"]], ["c"])


if __name__ == "__main__":
    unittest.main()
