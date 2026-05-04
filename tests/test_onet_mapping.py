import json
import shutil
import tempfile
import unittest
from pathlib import Path

from map_entities_to_onet import main as map_entities_main
from onet_mapping import OnetMapper, map_record_entities, prepare_onet_index, summarize_record_mappings
from prepare_onet_index import main as prepare_onet_index_main


def write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def build_onet_fixture(root: Path) -> Path:
    onet_dir = root / "onet"
    onet_dir.mkdir(parents=True, exist_ok=True)

    write_text(
        onet_dir / "Occupation Data.txt",
        """
O*NET-SOC Code\tTitle\tDescription
15-1252.00\tSoftware Developers\tDevelop and maintain backend and web applications.
15-2051.00\tData Scientists\tAnalyze datasets and build predictive models.
        """,
    )
    write_text(
        onet_dir / "Alternate Titles.txt",
        """
O*NET-SOC Code\tAlternate Title
15-1252.00\tBackend Developer
15-2051.00\tMachine Learning Engineer
        """,
    )
    write_text(
        onet_dir / "Skills.txt",
        """
O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tData Value
15-1252.00\t2.A.1\tProgramming\tIM\t4.75
15-1252.00\t2.A.1\tProgramming\tLV\t5.00
15-2051.00\t2.A.2\tData Analysis\tIM\t4.50
        """,
    )
    write_text(
        onet_dir / "Knowledge.txt",
        """
O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tData Value
15-1252.00\t2.C.1\tComputer Science\tIM\t4.85
        """,
    )
    write_text(
        onet_dir / "Abilities.txt",
        """
O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tData Value
15-1252.00\t1.A.1\tDeductive Reasoning\tIM\t4.00
        """,
    )
    write_text(
        onet_dir / "Work Activities.txt",
        """
O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tData Value
15-1252.00\t4.A.1\tDevelop and test software\tIM\t4.80
        """,
    )
    write_text(
        onet_dir / "Technology Skills.txt",
        """
O*NET-SOC Code\tCommodity Code\tCommodity Title\tExample\tHot Technology\tIn Demand
15-1252.00\t43230000\tProgramming Languages\tPython\tY\tY
15-2051.00\t43231513\tData Science Libraries\tPandas\tN\tY
        """,
    )
    write_text(
        onet_dir / "Task Statements.txt",
        """
O*NET-SOC Code\tTask ID\tTask
15-1252.00\t101\tDevelop REST APIs for backend services.
15-2051.00\t202\tAnalyze datasets and build machine learning models.
        """,
    )
    return onet_dir


class OnetMappingTests(unittest.TestCase):
    def test_prepare_onet_index_builds_expected_entry_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            onet_dir = build_onet_fixture(Path(tmp_dir))
            entries, summary = prepare_onet_index(onet_dir)

        self.assertEqual(summary["occupations"], 2)
        self.assertGreater(summary["entries"], 0)
        self.assertIn("occupation", summary["entry_types"])
        self.assertIn("alternate_title", summary["entry_types"])
        self.assertIn("technology_skill", summary["entry_types"])
        self.assertIn("task_statement", summary["entry_types"])
        self.assertTrue(any(entry.title == "Programming" for entry in entries))

    def test_mapper_links_role_skill_technology_and_work_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            onet_dir = build_onet_fixture(Path(tmp_dir))
            entries, _ = prepare_onet_index(onet_dir)
            mapper = OnetMapper(entries, embedding_model=None)
            row = {
                "text": "Backend Developer with Python and strong programming background.",
                "entities": [
                    {"label": "JOB_ROLE", "text": "Backend Developer", "start": 0, "end": 17},
                    {"label": "TECHNOLOGY", "text": "Python", "start": 23, "end": 29},
                    {"label": "SKILL", "text": "Programming", "start": 41, "end": 52},
                    {"label": "WORK_ACTIVITY", "text": "Develop REST APIs", "start": 53, "end": 70},
                    {"label": "INDUSTRY", "text": "Fintech", "start": 71, "end": 78},
                ],
            }

            mappings = map_record_entities(row, mapper, top_k=3, min_score=0.2)
            by_label = {mapping["entity_label"]: mapping for mapping in mappings}
            summary = summarize_record_mappings(mappings)

        self.assertEqual(
            by_label["JOB_ROLE"]["candidates"][0]["onetsoc_code"],
            "15-1252.00",
        )
        self.assertEqual(by_label["TECHNOLOGY"]["candidates"][0]["title"], "Python")
        self.assertEqual(by_label["SKILL"]["candidates"][0]["title"], "Programming")
        self.assertEqual(
            by_label["WORK_ACTIVITY"]["candidates"][0]["onetsoc_code"],
            "15-1252.00",
        )
        self.assertFalse(by_label["INDUSTRY"]["supported"])
        self.assertEqual(summary["role_onetsoc_codes"], ["15-1252.00"])
        self.assertGreater(summary["mapped_rate"], 0.0)

    def test_cli_pipeline_builds_index_and_maps_entities(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="onet_mapping_test_"))
        try:
            onet_dir = build_onet_fixture(root)
            input_path = root / "annotations.jsonl"
            index_path = root / "artifacts" / "onet_index.jsonl"
            output_path = root / "artifacts" / "mapped.jsonl"
            summary_path = root / "artifacts" / "mapped.summary.json"

            rows = [
                {
                    "id": "cv-1",
                    "doc_type": "cv",
                    "text": "Backend Developer using Python to develop REST APIs.",
                    "entities": [
                        {"label": "JOB_ROLE", "text": "Backend Developer", "start": 0, "end": 17},
                        {"label": "TECHNOLOGY", "text": "Python", "start": 24, "end": 30},
                        {"label": "WORK_ACTIVITY", "text": "develop REST APIs", "start": 34, "end": 51},
                    ],
                }
            ]
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            rc = prepare_onet_index_main(
                ["--onet-dir", str(onet_dir), "--output", str(index_path), "--overwrite"]
            )
            self.assertEqual(rc, 0)

            rc = map_entities_main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--onet-index",
                    str(index_path),
                    "--summary-output",
                    str(summary_path),
                    "--min-score",
                    "0.2",
                    "--overwrite",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())

            mapped_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(root)

        self.assertEqual(len(mapped_rows), 1)
        self.assertIn("onet_mappings", mapped_rows[0])
        self.assertIn("onet_mapping_summary", mapped_rows[0])
        self.assertEqual(summary["documents"], 1)
        self.assertEqual(summary["mapped_entities"], 3)


if __name__ == "__main__":
    unittest.main()
