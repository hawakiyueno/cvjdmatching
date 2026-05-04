import os
from pathlib import Path
import unittest

from djinni_openai_ner import (
    OpenAIChatCompletionsClient,
    build_openai_api_error,
    build_prompt,
    extract_message_text,
    load_api_keys,
    parse_retry_after_seconds,
    refine_annotation,
)


class DjinniOpenAINerTests(unittest.TestCase):
    def test_parse_retry_after_seconds(self) -> None:
        self.assertEqual(parse_retry_after_seconds("12"), 12.0)
        self.assertEqual(parse_retry_after_seconds(None), None)

    def test_build_openai_api_error_marks_quota(self) -> None:
        body = """
        {
          "error": {
            "message": "You exceeded your current quota.",
            "type": "insufficient_quota",
            "code": "insufficient_quota"
          }
        }
        """
        err = build_openai_api_error(429, body, {"Retry-After": "15"})
        self.assertTrue(err.quota_exhausted)
        self.assertTrue(err.rate_limited)
        self.assertEqual(err.retry_after_seconds, 15.0)

    def test_extract_message_text_supports_string_content(self) -> None:
        raw = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"document_language":"en","is_it_document":true,"summary":"","entities":[],"qualification_facts":[]}'
                    },
                }
            ]
        }
        self.assertIn('"document_language":"en"', extract_message_text(raw))

    def test_extract_message_text_supports_content_parts(self) -> None:
        raw = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": '{"document_language":"en","is_it_document":true,"summary":"","entities":[],"qualification_facts":[]}',
                            }
                        ]
                    },
                }
            ]
        }
        self.assertIn('"is_it_document":true', extract_message_text(raw))

    def test_build_prompt_mentions_thesis_goal_and_chunk(self) -> None:
        prompt = build_prompt("row-1", "cv", 0, 0, "Python developer with AWS")
        self.assertIn("Stage 1 of the thesis pipeline", prompt)
        self.assertIn("Python developer with AWS", prompt)

    def test_load_api_keys_prefers_env_and_file(self) -> None:
        path = Path("tmp_openai_keys.txt")
        original_key = os.environ.get("OPENAI_API_KEY")
        original_keys = os.environ.get("OPENAI_API_KEYS")
        try:
            path.write_text("file-key\n", encoding="utf-8")
            os.environ["OPENAI_API_KEY"] = "env-key"
            os.environ["OPENAI_API_KEYS"] = "multi-a,multi-b"
            args = type("Args", (), {"api_keys_file": path, "api_key": "cli-key"})
            self.assertEqual(load_api_keys(args), ["multi-a", "multi-b", "env-key", "file-key", "cli-key"])
        finally:
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key
            if original_keys is None:
                os.environ.pop("OPENAI_API_KEYS", None)
            else:
                os.environ["OPENAI_API_KEYS"] = original_keys
            if path.exists():
                path.unlink()

    def test_rotate_api_key_moves_to_next_slot(self) -> None:
        client = OpenAIChatCompletionsClient(api_keys=["key-a", "key-b"])
        err = build_openai_api_error(
            429,
            '{"error":{"message":"quota exceeded","type":"insufficient_quota","code":"insufficient_quota"}}',
        )
        self.assertTrue(client.rotate_api_key(err))
        self.assertEqual(client.active_key_index, 1)

    def test_refine_annotation_drops_ability_and_english_level_skill_noise(self) -> None:
        annotation = {
            "entities": [
                {"label": "SKILL", "text": "intermediate", "start": 0, "end": 12, "normalized": "intermediate"},
                {"label": "ABILITY", "text": "English Level: upper", "start": 13, "end": 33, "normalized": "english level: upper"},
                {"label": "SKILL", "text": "communication skills", "start": 34, "end": 54, "normalized": "communication skills"},
            ]
        }
        refined = refine_annotation(annotation)
        self.assertEqual(len(refined["entities"]), 1)
        self.assertEqual(refined["entities"][0]["text"], "communication skills")

    def test_refine_annotation_drops_skill_methodology_leakage(self) -> None:
        annotation = {
            "entities": [
                {"label": "SKILL", "text": "QA automation", "start": 0, "end": 13, "normalized": "qa automation"},
                {"label": "SKILL", "text": "Agile", "start": 14, "end": 19, "normalized": "agile"},
                {"label": "SKILL", "text": "design patterns", "start": 20, "end": 35, "normalized": "design patterns"},
                {"label": "SKILL", "text": "MVVM", "start": 36, "end": 40, "normalized": "mvvm"},
                {"label": "SKILL", "text": "communication skills", "start": 41, "end": 61, "normalized": "communication skills"},
            ]
        }
        refined = refine_annotation(annotation)
        self.assertEqual(len(refined["entities"]), 1)
        self.assertEqual(refined["entities"][0]["text"], "communication skills")

    def test_refine_annotation_keeps_testing_and_data_domain_skills(self) -> None:
        annotation = {
            "entities": [
                {"label": "SKILL", "text": "manual testing", "start": 0, "end": 14, "normalized": "manual testing"},
                {"label": "SKILL", "text": "test automation", "start": 15, "end": 30, "normalized": "test automation"},
                {"label": "SKILL", "text": "data science", "start": 31, "end": 43, "normalized": "data science"},
            ]
        }
        refined = refine_annotation(annotation)
        self.assertEqual([entity["text"] for entity in refined["entities"]], ["manual testing", "test automation", "data science"])

    def test_refine_annotation_drops_degree_field_only_text(self) -> None:
        annotation = {
            "entities": [
                {"label": "DEGREE", "text": "computer science", "start": 0, "end": 16, "normalized": "computer science"},
                {"label": "DEGREE", "text": "Bachelor's degree in Computer Science", "start": 17, "end": 55, "normalized": "bachelor's degree in computer science"},
            ]
        }
        refined = refine_annotation(annotation)
        self.assertEqual(len(refined["entities"]), 1)
        self.assertEqual(refined["entities"][0]["text"], "Bachelor's degree in Computer Science")

    def test_refine_annotation_drops_generic_project_type_and_non_industry(self) -> None:
        annotation = {
            "entities": [
                {"label": "PROJECT_TYPE", "text": "application", "start": 0, "end": 11, "normalized": "application"},
                {"label": "PROJECT_TYPE", "text": "web application", "start": 12, "end": 27, "normalized": "web application"},
                {"label": "PROJECT_TYPE", "text": "mobile app", "start": 28, "end": 38, "normalized": "mobile app"},
                {"label": "PROJECT_TYPE", "text": "pet project", "start": 39, "end": 50, "normalized": "pet project"},
                {"label": "PROJECT_TYPE", "text": "CRM system", "start": 51, "end": 61, "normalized": "crm system"},
                {"label": "INDUSTRY", "text": "IT", "start": 62, "end": 64, "normalized": "it"},
                {"label": "INDUSTRY", "text": "product company", "start": 65, "end": 80, "normalized": "product company"},
                {"label": "INDUSTRY", "text": "technology", "start": 81, "end": 91, "normalized": "technology"},
                {"label": "INDUSTRY", "text": "game development", "start": 92, "end": 108, "normalized": "game development"},
                {"label": "INDUSTRY", "text": "computer vision", "start": 109, "end": 124, "normalized": "computer vision"},
                {"label": "INDUSTRY", "text": "fintech", "start": 125, "end": 132, "normalized": "fintech"},
            ]
        }
        refined = refine_annotation(annotation)
        self.assertEqual([entity["text"] for entity in refined["entities"]], ["CRM system", "fintech"])

    def test_refine_annotation_drops_generic_work_activity(self) -> None:
        annotation = {
            "entities": [
                {"label": "WORK_ACTIVITY", "text": "testing", "start": 0, "end": 7, "normalized": "testing"},
                {"label": "WORK_ACTIVITY", "text": "code review", "start": 8, "end": 19, "normalized": "code review"},
                {"label": "WORK_ACTIVITY", "text": "build ETL pipelines", "start": 20, "end": 39, "normalized": "build etl pipelines"},
            ]
        }
        refined = refine_annotation(annotation)
        self.assertEqual(len(refined["entities"]), 1)
        self.assertEqual(refined["entities"][0]["text"], "build ETL pipelines")

    def test_refine_annotation_drops_degree_noise_and_keeps_real_credentials(self) -> None:
        annotation = {
            "entities": [
                {"label": "DEGREE", "text": "Bachelor thesis", "start": 0, "end": 16, "normalized": "bachelor thesis"},
                {"label": "DEGREE", "text": "PhD student", "start": 17, "end": 28, "normalized": "phd student"},
                {"label": "DEGREE", "text": "computer science degree", "start": 29, "end": 52, "normalized": "computer science degree"},
            ]
        }
        refined = refine_annotation(annotation)
        self.assertEqual([entity["text"] for entity in refined["entities"]], ["computer science degree"])

    def test_refine_annotation_drops_certification_noise_and_keeps_named_certs(self) -> None:
        annotation = {
            "entities": [
                {"label": "CERTIFICATION", "text": "AWS", "start": 0, "end": 3, "normalized": "aws"},
                {"label": "CERTIFICATION", "text": "Hillel IT School", "start": 4, "end": 20, "normalized": "hillel it school"},
                {"label": "CERTIFICATION", "text": "training program", "start": 21, "end": 37, "normalized": "training program"},
                {"label": "CERTIFICATION", "text": "AWS Certified Solutions Architect - Associate", "start": 38, "end": 84, "normalized": "aws certified solutions architect - associate"},
                {"label": "CERTIFICATION", "text": "Oracle Certified Associate", "start": 85, "end": 112, "normalized": "oracle certified associate"},
            ]
        }
        refined = refine_annotation(annotation)
        self.assertEqual(
            [entity["text"] for entity in refined["entities"]],
            ["AWS Certified Solutions Architect - Associate", "Oracle Certified Associate"],
        )


if __name__ == "__main__":
    unittest.main()
