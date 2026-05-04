import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_hf_recruitment_subset import (
    count_existing_rows,
    compose_cv_text,
    compose_jd_text,
    get_total_rows,
    inspect_existing_rows,
    iter_dataset_rows,
    iter_normalized_rows,
    normalize_cv_row,
    normalize_jd_row,
)


class PrepareHfRecruitmentSubsetTests(unittest.TestCase):
    def test_compose_cv_text_prefers_cv_field(self) -> None:
        row = {
            "Position": "Data Scientist",
            "Primary Keyword": "Data Science",
            "English Level": "upper-intermediate",
            "Experience Years": 4,
            "CV": "Built machine learning pipelines with Python, SQL, TensorFlow, and AWS for production analytics systems.",
            "Moreinfo": "Extra details",
        }
        text = compose_cv_text(row)
        self.assertIn("Position: Data Scientist", text)
        self.assertIn("CV:\nBuilt machine learning pipelines with Python, SQL, TensorFlow, and AWS for production analytics systems.", text)
        self.assertNotIn("Extra details", text)

    def test_compose_cv_text_falls_back_to_extra_sections(self) -> None:
        row = {
            "Position": "Backend Engineer",
            "CV": "Short",
            "Moreinfo": "Worked with Django and PostgreSQL.",
            "Looking For": "Remote backend role.",
        }
        text = compose_cv_text(row)
        self.assertIn("More Info:\nWorked with Django and PostgreSQL.", text)
        self.assertIn("Looking For:\nRemote backend role.", text)

    def test_normalize_cv_row_prefixes_id(self) -> None:
        row = {
            "id": "abc",
            "CV": "Python developer",
            "CV_lang": "en",
        }
        normalized = normalize_cv_row(row)
        self.assertEqual(normalized["id"], "cv::abc")
        self.assertEqual(normalized["doc_type"], "cv")

    def test_compose_jd_text_includes_position_and_description(self) -> None:
        row = {
            "Position": "ML Engineer",
            "Company Name": "Acme",
            "Long Description": "Need Python, AWS, and 3 years experience.",
            "Exp Years": "3y",
        }
        text = compose_jd_text(row)
        self.assertIn("Position: ML Engineer", text)
        self.assertIn("Company Name: Acme", text)
        self.assertIn("Job Description:\nNeed Python, AWS, and 3 years experience.", text)

    def test_normalize_jd_row_prefixes_id(self) -> None:
        row = {
            "id": "xyz",
            "Long Description": "JD body",
            "Long Description_lang": "en",
        }
        normalized = normalize_jd_row(row)
        self.assertEqual(normalized["id"], "jd::xyz")
        self.assertEqual(normalized["doc_type"], "jd")

    def test_count_existing_rows(self) -> None:
        path = Path("tmp_subset_counts.jsonl")
        try:
            path.write_text(
                '{"doc_type":"cv"}\n{"doc_type":"jd"}\n{"doc_type":"cv"}\n',
                encoding="utf-8",
            )
            self.assertEqual(count_existing_rows(path), (2, 1))
        finally:
            if path.exists():
                path.unlink()

    def test_inspect_existing_rows_reads_source_offsets(self) -> None:
        path = Path("tmp_subset_offsets.jsonl")
        try:
            path.write_text(
                "\n".join(
                    [
                        '{"doc_type":"cv","source_offset":10}',
                        '{"doc_type":"cv","source_offset":15}',
                        '{"doc_type":"jd","source_offset":3}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(inspect_existing_rows(path), (2, 1, 16, 4))
        finally:
            if path.exists():
                path.unlink()

    def test_get_total_rows_returns_count(self) -> None:
        total = get_total_rows("lang-uk/recruitment-dataset-candidate-profiles-english", cache_dir=None, token=None)
        self.assertIsInstance(total, int)
        self.assertGreater(total, 0)

    def test_iter_dataset_rows_returns_first_row(self) -> None:
        rows = list(iter_dataset_rows("lang-uk/recruitment-dataset-job-descriptions-english", offset=0, limit=1, cache_dir=None, token=None))
        self.assertEqual(len(rows), 1)
        self.assertIn("Long Description", rows[0])

    def test_iter_normalized_rows_scans_past_non_it_rows(self) -> None:
        raw_rows = [
            {
                "id": "1",
                "Position": "Business Analyst",
                "Primary Keyword": "Business Analyst",
                "CV": "Led workshops and stakeholder reporting without technical delivery focus.",
                "CV_lang": "en",
            },
            {
                "id": "2",
                "Position": "Python Developer",
                "Primary Keyword": "Python",
                "CV": "Built backend services with Python, SQL, Docker, and AWS across production systems.",
                "CV_lang": "en",
            },
        ]
        with patch("prepare_hf_recruitment_subset.get_total_rows", return_value=2), patch(
            "prepare_hf_recruitment_subset.iter_dataset_rows",
            return_value=iter(raw_rows),
        ):
            rows = list(
                iter_normalized_rows(
                    dataset_name="dummy",
                    offset=0,
                    limit=1,
                    cache_dir=None,
                    token=None,
                    normalizer=normalize_cv_row,
                    it_only=True,
                )
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "cv::2")
        self.assertEqual(rows[0]["source_offset"], 1)

    def test_iter_normalized_rows_strict_skips_ambiguous_ba_and_keeps_other_developer(self) -> None:
        raw_rows = [
            {
                "id": "1",
                "Position": "Business Analyst",
                "Primary Keyword": "Business Analyst",
                "CV": "Worked with SQL, Python, dashboards, and ETL pipelines.",
                "CV_lang": "en",
            },
            {
                "id": "2",
                "Position": "1C Developer",
                "Primary Keyword": "Other",
                "CV": "Built backend modules with 1C Enterprise, SQL, and integrations across ERP systems.",
                "CV_lang": "en",
            },
        ]
        with patch("prepare_hf_recruitment_subset.get_total_rows", return_value=2), patch(
            "prepare_hf_recruitment_subset.iter_dataset_rows",
            return_value=iter(raw_rows),
        ):
            rows = list(
                iter_normalized_rows(
                    dataset_name="dummy",
                    offset=0,
                    limit=1,
                    cache_dir=None,
                    token=None,
                    normalizer=normalize_cv_row,
                    it_only=True,
                    it_strict=True,
                )
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "cv::2")
        self.assertEqual(rows[0]["source_offset"], 1)


if __name__ == "__main__":
    unittest.main()
