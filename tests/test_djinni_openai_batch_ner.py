import json
import shutil
import unittest
from pathlib import Path

from djinni_openai_batch_ner import (
    build_batch_request_line,
    finalize_batches,
    parse_batch_output_body,
    prepare_batch_workdir,
    retarget_pending_shards_model,
)


class DjinniOpenAIBatchNerTests(unittest.TestCase):
    def test_build_batch_request_line_targets_chat_completions(self) -> None:
        line = build_batch_request_line("chunk-1", "gpt-4.1-mini", "hello")
        row = json.loads(line.decode("utf-8"))
        self.assertEqual(row["custom_id"], "chunk-1")
        self.assertEqual(row["url"], "/v1/chat/completions")
        self.assertEqual(row["body"]["response_format"]["type"], "json_schema")

    def test_parse_batch_output_body_reads_nested_chat_completion(self) -> None:
        row = {
            "custom_id": "chunk-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": '{"document_language":"en","is_it_document":true,"summary":"Python backend profile.","entities":[],"qualification_facts":[]}'
                            },
                        }
                    ]
                },
            },
        }
        parsed = parse_batch_output_body(row)
        self.assertEqual(parsed["document_language"], "en")
        self.assertTrue(parsed["is_it_document"])

    def test_prepare_and_finalize_batch_workdir(self) -> None:
        root = Path("tmp_openai_batch_test")
        input_path = root / "input.jsonl"
        output_path = root / "final.jsonl"
        try:
            root.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "id": "cv::1",
                    "doc_type": "cv",
                    "source_language": "en",
                    "text": "Position: Backend Engineer\n\nCV:\nBuilt backend services with Python and PostgreSQL.",
                },
                {
                    "id": "jd::2",
                    "doc_type": "jd",
                    "source_language": "en",
                    "text": "Position: Data Engineer\n\nJob Description:\nBuild ETL pipelines with Python, SQL, and Airflow.",
                },
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            class PrepareArgs:
                input = input_path
                work_dir = root / "work"
                model = "gpt-4.1-mini"
                id_column = "id"
                text_column = "text"
                text_columns = None
                document_type = "auto"
                doc_type_column = "doc_type"
                language_column = "source_language"
                max_records = None
                max_documents = None
                max_chars_per_call = 12000
                chunk_overlap = 200
                max_file_bytes = 1_000_000
                max_requests_per_shard = 100
                max_estimated_tokens_per_shard = 1_000_000
                overwrite = True
                resume_output = None
                verbose = False

            manifest = prepare_batch_workdir(PrepareArgs)
            self.assertEqual(len(manifest["shards"]), 1)

            work_dir = PrepareArgs.work_dir
            jobs = {
                "jobs": [
                    {
                        "shard_id": "shard-00001",
                        "local_output_path": "outputs/shard-00001.output.jsonl",
                        "batch": {"id": "batch_123", "status": "completed"},
                    }
                ]
            }
            (work_dir / "outputs").mkdir(parents=True, exist_ok=True)
            (work_dir / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")

            chunk_rows = []
            with (work_dir / "chunks" / "shard-00001.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    chunk_rows.append(json.loads(line))

            output_rows = []
            for chunk in chunk_rows:
                text = chunk["chunk_text"]
                if "Backend Engineer" in text:
                    body = {
                        "document_language": "en",
                        "is_it_document": True,
                        "summary": "Backend engineer CV.",
                        "entities": [
                            {"label": "JOB_ROLE", "text": "Backend Engineer", "start": 10, "end": 26, "normalized": "backend engineer"},
                            {"label": "TECHNOLOGY", "text": "Python", "start": 55, "end": 61, "normalized": "python"},
                        ],
                        "qualification_facts": [],
                    }
                else:
                    body = {
                        "document_language": "en",
                        "is_it_document": True,
                        "summary": "Data engineer JD.",
                        "entities": [
                            {"label": "JOB_ROLE", "text": "Data Engineer", "start": 10, "end": 23, "normalized": "data engineer"},
                            {"label": "TECHNOLOGY", "text": "Airflow", "start": 80, "end": 87, "normalized": "airflow"},
                        ],
                        "qualification_facts": [],
                    }
                output_rows.append(
                    {
                        "custom_id": chunk["custom_id"],
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "finish_reason": "stop",
                                        "message": {"content": json.dumps(body)},
                                    }
                                ]
                            },
                        },
                    }
                )
            (work_dir / "outputs" / "shard-00001.output.jsonl").write_text(
                "\n".join(json.dumps(row) for row in output_rows) + "\n",
                encoding="utf-8",
            )

            finalize_args = type(
                "FinalizeArgs",
                (),
                {
                    "work_dir": work_dir,
                    "output": output_path,
                    "resume": False,
                    "require_english": True,
                    "require_it": True,
                    "verbose": False,
                },
            )

            rc = finalize_batches(finalize_args)
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            labels = {entity["label"] for row in rows for entity in row["entities"]}
            self.assertIn("JOB_ROLE", labels)
            self.assertIn("TECHNOLOGY", labels)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_retarget_model_updates_only_pending_shards(self) -> None:
        root = Path("tmp_openai_batch_retarget_test")
        input_path = root / "input.jsonl"
        try:
            root.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "id": "cv::1",
                    "doc_type": "cv",
                    "source_language": "en",
                    "text": "Position: Backend Engineer\n\nCV:\nBuilt backend services with Python and PostgreSQL.",
                },
                {
                    "id": "jd::2",
                    "doc_type": "jd",
                    "source_language": "en",
                    "text": "Position: Data Engineer\n\nJob Description:\nBuild ETL pipelines with Python, SQL, and Airflow.",
                },
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            class PrepareArgs:
                input = input_path
                work_dir = root / "work"
                model = "gpt-4.1-mini"
                id_column = "id"
                text_column = "text"
                text_columns = None
                document_type = "auto"
                doc_type_column = "doc_type"
                language_column = "source_language"
                max_records = None
                max_documents = None
                max_chars_per_call = 12000
                chunk_overlap = 200
                max_file_bytes = 1_000_000
                max_requests_per_shard = 1
                max_estimated_tokens_per_shard = 1_000_000
                overwrite = True
                resume_output = None
                verbose = False

            manifest = prepare_batch_workdir(PrepareArgs)
            self.assertEqual(len(manifest["shards"]), 2)

            jobs = {
                "jobs": [
                    {
                        "shard_id": "shard-00001",
                        "batch": {
                            "id": "batch_123",
                            "status": "completed",
                            "model": "gpt-4.1-mini-2025-04-14",
                        },
                    }
                ]
            }
            jobs_path = PrepareArgs.work_dir / "jobs.json"
            jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")

            retarget_args = type(
                "RetargetArgs",
                (),
                {
                    "work_dir": PrepareArgs.work_dir,
                    "model": "gpt-4o-mini",
                    "verbose": False,
                },
            )
            updated_manifest = retarget_pending_shards_model(retarget_args)
            self.assertEqual(updated_manifest["model"], "mixed")
            self.assertEqual(updated_manifest["pending_model"], "gpt-4o-mini")

            request_1 = json.loads((PrepareArgs.work_dir / "requests" / "shard-00001.jsonl").read_text(encoding="utf-8").splitlines()[0])
            request_2 = json.loads((PrepareArgs.work_dir / "requests" / "shard-00002.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(request_1["body"]["model"], "gpt-4.1-mini")
            self.assertEqual(request_2["body"]["model"], "gpt-4o-mini")

            records = [
                json.loads(line)
                for line in (PrepareArgs.work_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["model"] for row in records], ["gpt-4.1-mini", "gpt-4o-mini"])
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
