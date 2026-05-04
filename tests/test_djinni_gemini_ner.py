from pathlib import Path
import unittest

from djinni_gemini_ner import (
    GeminiRestClient,
    build_gemini_api_error,
    canonicalize_operator,
    dedupe_entities,
    infer_doc_type,
    merge_api_keys,
    normalize_entity_label,
    parse_retry_delay_seconds,
    repair_offsets,
    sanitize_entity,
    split_text,
    load_api_keys_from_file,
)


class DjinniGeminiNerTests(unittest.TestCase):
    def test_repair_offsets_uses_exact_match(self) -> None:
        text = "Python, SQL, and Python again"
        repaired = repair_offsets(text, "Python", 18, 24)
        self.assertEqual(repaired, (17, 23))

    def test_sanitize_entity_normalizes_label_and_shift(self) -> None:
        chunk = "Built ETL pipelines with Python and Airflow."
        entity = {
            "label": "technologies",
            "text": "Python",
            "start": 25,
            "end": 31,
            "normalized": "",
        }
        cleaned = sanitize_entity(chunk, entity, shift=100)
        self.assertEqual(cleaned["label"], "TECHNOLOGY")
        self.assertEqual(cleaned["start"], 125)
        self.assertEqual(cleaned["end"], 131)

    def test_split_text_keeps_offsets_sorted(self) -> None:
        text = ("Paragraph one.\n\n" * 20).strip()
        chunks = split_text(text, max_chars=60, overlap=10)
        self.assertGreater(len(chunks), 1)
        starts = [start for start, _ in chunks]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(chunk for _, chunk in chunks))

    def test_infer_doc_type_prefers_metadata(self) -> None:
        self.assertEqual(infer_doc_type("resume", "jd", "job_description"), "cv")
        self.assertEqual(infer_doc_type("", "auto", "job_description"), "jd")

    def test_dedupe_entities_merges_same_span(self) -> None:
        entities = [
            {"label": "TECHNOLOGY", "text": "Python", "start": 0, "end": 6, "normalized": "python"},
            {"label": "TECHNOLOGY", "text": "Python", "start": 0, "end": 6, "normalized": "python"},
        ]
        self.assertEqual(len(dedupe_entities(entities)), 1)

    def test_normalize_entity_label_rejects_unknown(self) -> None:
        self.assertIsNone(normalize_entity_label("salary"))

    def test_canonicalize_operator_falls_back_for_cv_credentials(self) -> None:
        self.assertEqual(canonicalize_operator("weird-token", "DEGREE", False), "=")
        self.assertEqual(canonicalize_operator("weird-token", "EXPERIENCE_YEARS", False), "=")

    def test_parse_retry_delay_seconds(self) -> None:
        self.assertEqual(parse_retry_delay_seconds("31s"), 31.0)
        self.assertEqual(parse_retry_delay_seconds("1500ms"), 1.5)

    def test_build_gemini_api_error_marks_quota(self) -> None:
        body = """
        {
          "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded",
            "details": [{"retryDelay": "31s"}]
          }
        }
        """
        err = build_gemini_api_error(429, body)
        self.assertTrue(err.quota_exhausted)
        self.assertEqual(err.retry_after_seconds, 31.0)

    def test_merge_api_keys_deduplicates(self) -> None:
        keys = merge_api_keys(["a", "a", " b ", "", "c"])
        self.assertEqual(keys, ["a", "b", "c"])

    def test_load_api_keys_from_file(self) -> None:
        path = Path("tmp_gemini_keys.txt")
        try:
            path.write_text("key1\n\nkey2\nkey1\n", encoding="utf-8")
            self.assertEqual(load_api_keys_from_file(path), ["key1", "key2"])
        finally:
            if path.exists():
                path.unlink()

    def test_rotate_api_key_moves_to_next_slot(self) -> None:
        client = GeminiRestClient(api_keys=["key-a", "key-b"])
        err = build_gemini_api_error(429, '{"error":{"status":"RESOURCE_EXHAUSTED","message":"quota exceeded"}}')
        self.assertTrue(client.rotate_api_key(err))
        self.assertEqual(client.active_key_index, 1)


if __name__ == "__main__":
    unittest.main()
