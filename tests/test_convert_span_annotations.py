import unittest

from convert_span_annotations import build_output_row, extract_entities


class ConvertSpanAnnotationsTests(unittest.TestCase):
    def test_extract_entities_reads_doccano_labels(self) -> None:
        text = "Python developer with AWS"
        row = {
            "text": text,
            "labels": [
                [0, 6, "TECHNOLOGY"],
                [7, 16, "JOB_ROLE"],
                [22, 25, "TECHNOLOGY"],
            ],
        }
        entities = extract_entities(text, row, {})
        self.assertEqual([entity["label"] for entity in entities], ["TECHNOLOGY", "JOB_ROLE", "TECHNOLOGY"])
        self.assertEqual(entities[0]["text"], "Python")

    def test_extract_entities_applies_label_map(self) -> None:
        text = "SQL reporting"
        row = {
            "text": text,
            "annotations": [
                {"start": 0, "end": 3, "label": "Skills"},
                {"start": 4, "end": 13, "entity": "ABILITY"},
            ],
        }
        entities = extract_entities(text, row, {"Skills": "TECHNOLOGY"})
        self.assertEqual([entity["label"] for entity in entities], ["TECHNOLOGY"])

    def test_extract_entities_maps_prefixed_public_labels(self) -> None:
        text = "Python developer"
        row = {
            "text": text,
            "labels": [
                [0, 6, "SKILL: python"],
                [7, 16, "Occupation"],
            ],
        }
        entities = extract_entities(text, row, {})
        self.assertEqual([entity["label"] for entity in entities], ["SKILL", "JOB_ROLE"])

    def test_build_output_row_uses_defaults(self) -> None:
        row = {
            "doc_id": "abc",
            "text": "React engineer",
            "labels": [[0, 5, "TECHNOLOGY"], [6, 14, "JOB_ROLE"]],
        }
        output = build_output_row(
            row=row,
            index=1,
            text_field="text",
            id_field="doc_id",
            doc_type_field=None,
            default_doc_type="jd",
            label_map={},
        )
        self.assertIsNotNone(output)
        assert output is not None
        self.assertEqual(output["record_id"], "abc")
        self.assertEqual(output["document_type"], "jd")
        self.assertEqual(len(output["entities"]), 2)


if __name__ == "__main__":
    unittest.main()
