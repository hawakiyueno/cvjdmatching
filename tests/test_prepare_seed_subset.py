import json
import unittest
from pathlib import Path

from prepare_seed_subset import build_seed_subset, clean_text, count_words, is_it_row, length_bucket, load_and_clean_rows


class PrepareSeedSubsetTests(unittest.TestCase):
    def test_clean_text_normalizes_common_noise(self) -> None:
        raw = "Line 1\r\n\r\n\r\nLine 2â€™s text\u00a0with Â extra  spaces"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Line 1\n\nLine 2's text with extra spaces")

    def test_count_words_handles_skills(self) -> None:
        self.assertGreaterEqual(count_words("Python C# Node.js ASP.NET"), 4)

    def test_length_bucket_is_doc_type_specific(self) -> None:
        self.assertEqual(length_bucket("cv", 300), "short")
        self.assertEqual(length_bucket("jd", 300), "short")
        self.assertEqual(length_bucket("jd", 2500), "long")

    def test_load_and_clean_rows_filters_sparse_records(self) -> None:
        path = Path("tmp_seed_input.jsonl")
        try:
            rows = [
                {
                    "id": "cv::1",
                    "doc_type": "cv",
                    "source_language": "en",
                    "text": "Position: Engineer\n\nCV:\nBuilt APIs with Python, SQL, Docker, and AWS across multiple backend services for enterprise systems.",
                    "primary_keyword": "Python",
                },
                {
                    "id": "cv::2",
                    "doc_type": "cv",
                    "source_language": "en",
                    "text": "Too short",
                    "primary_keyword": "Python",
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            class Args:
                input = path
                min_cv_chars = 50
                min_jd_chars = 50
                min_cv_words = 8
                min_jd_words = 8
                it_only = False

            cleaned, stats = load_and_clean_rows(Args)
            self.assertEqual(len(cleaned), 1)
            self.assertEqual(stats["dropped_short_or_sparse"], 1)
        finally:
            if path.exists():
                path.unlink()

    def test_build_seed_subset_balances_types(self) -> None:
        rows = []
        for idx in range(3):
            rows.append(
                {
                    "id": f"cv::{idx}",
                    "doc_type": "cv",
                    "text": "x" * (500 + idx),
                    "primary_keyword": f"kw{idx}",
                    "length_bucket": "medium",
                }
            )
            rows.append(
                {
                    "id": f"jd::{idx}",
                    "doc_type": "jd",
                    "text": "y" * (1500 + idx),
                    "primary_keyword": f"kw{idx}",
                    "length_bucket": "medium",
                }
            )

        class Args:
            seed_cv = 2
            seed_jd = 2
            random_seed = 7

        seed_rows, stats = build_seed_subset(rows, Args)
        self.assertEqual(len(seed_rows), 4)
        self.assertEqual(stats["seed_by_type"], {"cv": 2, "jd": 2})

    def test_is_it_row_uses_keyword_and_text_signals(self) -> None:
        row1 = {"primary_keyword": "Python", "position": "Engineer"}
        self.assertTrue(is_it_row(row1, "Some cleaned text"))

        row2 = {"primary_keyword": "Business Analyst", "position": "Business Analyst"}
        self.assertFalse(is_it_row(row2, "Prepared reporting and stakeholder communication"))
        self.assertTrue(is_it_row(row2, "Worked with SQL, Python, dashboards, and ETL pipelines"))

    def test_is_it_row_strict_rejects_ambiguous_business_roles(self) -> None:
        row = {"primary_keyword": "Business Analyst", "position": "Business Analyst"}
        self.assertFalse(is_it_row(row, "Worked with SQL, Python, dashboards, and ETL pipelines", strict=True))

    def test_is_it_row_strict_keeps_other_when_position_is_clearly_technical(self) -> None:
        row = {"primary_keyword": "Other", "position": "1C Developer"}
        text = "Built integrations in 1C Enterprise, SQL, and backend services for internal systems."
        self.assertTrue(is_it_row(row, text, strict=True))


if __name__ == "__main__":
    unittest.main()
