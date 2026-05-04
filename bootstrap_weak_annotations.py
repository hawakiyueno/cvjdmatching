from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ENTITY_LABELS = {
    "TECHNOLOGY",
    "JOB_ROLE",
    "SKILL",
    "WORK_ACTIVITY",
    "INDUSTRY",
    "PROJECT_TYPE",
    "DEGREE",
    "CERTIFICATION",
}

ROLE_KEYWORDS = {
    "Business Analyst",
    "BI Developer",
    "Data Analyst",
    "Data Engineer",
    "Data Scientist",
    "Data Architect",
    "Solutions Architect",
    "Solution Architect",
    "Product Manager",
    "Project Manager",
    "Scrum Master",
    "Team Lead",
    "Tech Lead",
    "QA Automation Engineer",
    "QA Engineer",
    "QA Automation",
    "DevOps Engineer",
    "Support Engineer",
    "Support Analyst",
    "Customer Support Specialist",
    "Technical Writer",
    "Sysadmin",
    "System Administrator",
    "Security Engineer",
    "Frontend Developer",
    "Front-end Developer",
    "Backend Developer",
    "Back-end Developer",
    "Full Stack Developer",
    "Fullstack Developer",
    "Software Engineer",
    "Android Developer",
    "iOS Developer",
    "Unity Developer",
    "Java Developer",
    "Python Developer",
    "PHP Developer",
    ".NET Developer",
    "Node.js Developer",
    "Golang Developer",
    "Ruby Developer",
    "Scala Developer",
    "C++ Developer",
}

ROLE_HINT_TOKENS = (
    "developer",
    "engineer",
    "analyst",
    "manager",
    "writer",
    "architect",
    "administrator",
    "admin",
    "scientist",
    "designer",
    "consultant",
    "specialist",
)

GENERIC_ROLE_TERMS = {
    "analyst",
    "architect",
    "developer",
    "engineer",
    "lead",
    "manager",
    "qa",
    "support",
    "writer",
}

TECHNOLOGY_TERMS = {
    ".NET",
    "ASP.NET",
    "Airflow",
    "Android",
    "Angular",
    "Ansible",
    "AWS",
    "Azure",
    "Bitbucket",
    "C#",
    "C++",
    "CSS",
    "Cypress",
    "Django",
    "Docker",
    "Elasticsearch",
    "Express",
    "FastAPI",
    "Firebase",
    "Flask",
    "GCP",
    "Git",
    "GitHub",
    "GitLab",
    "Go",
    "Golang",
    "Grafana",
    "GraphQL",
    "Helm",
    "HTML",
    "Java",
    "JavaScript",
    "Jenkins",
    "Jira",
    "Kafka",
    "Kotlin",
    "Kubernetes",
    "Laravel",
    "Linux",
    "MongoDB",
    "MySQL",
    "Node.js",
    "NoSQL",
    "NumPy",
    "Pandas",
    "PHP",
    "Playwright",
    "PostgreSQL",
    "Postgres",
    "Power BI",
    "PowerBI",
    "Prometheus",
    "PyTorch",
    "Python",
    "React",
    "Redis",
    "REST API",
    "Ruby",
    "Rust",
    "Scala",
    "scikit-learn",
    "Selenium",
    "Spark",
    "Spring",
    "Spring Boot",
    "SQL",
    "Swift",
    "Tableau",
    "Terraform",
    "TensorFlow",
    "TypeScript",
    "Unity",
    "Unreal Engine",
    "Vue",
}

SKILL_TERMS = {
    "analytical skills",
    "communication",
    "communication skills",
    "critical thinking",
    "mentoring",
    "negotiation",
    "presentation skills",
    "product management",
    "project management",
    "requirements analysis",
    "stakeholder management",
    "teamwork",
    "time management",
    "troubleshooting",
    "writing skills",
}

INDUSTRY_TERMS = {
    "adtech",
    "automotive",
    "banking",
    "blockchain",
    "crypto",
    "cybersecurity",
    "e-commerce",
    "ecommerce",
    "edtech",
    "fintech",
    "gaming",
    "healthcare",
    "insurance",
    "martech",
    "telecom",
    "web3",
}

PROJECT_TYPE_TERMS = {
    "api platform",
    "blockchain project",
    "cloud platform",
    "computer vision",
    "dashboard",
    "data pipeline",
    "data warehouse",
    "etl pipeline",
    "machine learning model",
    "microservices",
    "mobile application",
    "mobile app",
    "recommender system",
    "saas product",
    "web application",
    "web app",
}

CERTIFICATION_PATTERNS = (
    r"\bAWS Certified(?: [A-Za-z][A-Za-z -]+)?\b",
    r"\bAzure (?:Administrator|Fundamentals|Solutions Architect|Developer)(?: Associate| Expert)?\b",
    r"\bGoogle Cloud (?:Professional|Associate) [A-Za-z -]+\b",
    r"\bISTQB\b",
    r"\bPMP\b",
    r"\bCertified Kubernetes Administrator\b",
    r"\bCKA\b",
    r"\bCCNA\b",
)

DEGREE_PATTERNS = (
    r"\bBachelor's(?: degree)?(?: in [A-Za-z][A-Za-z /&-]+| of [A-Za-z][A-Za-z /&-]+)?\b",
    r"\bBachelor degree(?: in [A-Za-z][A-Za-z /&-]+| of [A-Za-z][A-Za-z /&-]+)?\b",
    r"\bBachelor of [A-Za-z][A-Za-z /&-]+\b",
    r"\bBachelor in [A-Za-z][A-Za-z /&-]+\b",
    r"\bMaster's(?: degree)?(?: in [A-Za-z][A-Za-z /&-]+| of [A-Za-z][A-Za-z /&-]+)?\b",
    r"\bMaster degree(?: in [A-Za-z][A-Za-z /&-]+| of [A-Za-z][A-Za-z /&-]+)?\b",
    r"\bMaster of [A-Za-z][A-Za-z /&-]+\b",
    r"\bMaster in [A-Za-z][A-Za-z /&-]+\b",
    r"\bPh\.?D\.?(?: in [A-Za-z][A-Za-z /&-]+)?\b",
    r"\bB\.?Sc\.?(?: in [A-Za-z][A-Za-z /&-]+)?\b",
    r"\bM\.?Sc\.?(?: in [A-Za-z][A-Za-z /&-]+)?\b",
    r"\bMBA\b",
)

WORK_ACTIVITY_VERBS = (
    "analyze",
    "automate",
    "build",
    "collaborate",
    "conduct",
    "coordinate",
    "create",
    "deliver",
    "deploy",
    "design",
    "develop",
    "document",
    "estimate",
    "ghostwrote",
    "help",
    "implement",
    "integrate",
    "lead",
    "maintain",
    "manage",
    "migrate",
    "monitor",
    "optimize",
    "participate",
    "plan",
    "produce",
    "research",
    "support",
    "take part",
    "test",
    "troubleshoot",
    "write",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weak span annotations from the recruitment seed subset.")
    parser.add_argument("--input", type=Path, default=Path("hf_recruitment_seed_it_1000.jsonl"))
    parser.add_argument("--cv-output", type=Path, default=Path("doccano_export_cv.jsonl"))
    parser.add_argument("--jd-output", type=Path, default=Path("doccano_export_jd.jsonl"))
    parser.add_argument("--combined-output", type=Path, default=Path("djinni_ner_annotations.jsonl"))
    parser.add_argument("--stats-output", type=Path, default=Path("bootstrap_weak_annotations_stats.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} already exists. Pass --overwrite to replace it.")


def boundary_pattern(term: str) -> str:
    escaped = re.escape(term)
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def add_match(spans: dict[tuple[str, int, int], dict[str, Any]], text: str, label: str, start: int, end: int) -> None:
    if label not in ENTITY_LABELS or start < 0 or end <= start or end > len(text):
        return
    span_text = text[start:end]
    key = (label, start, end)
    spans[key] = {
        "label": label,
        "text": span_text,
        "start": start,
        "end": end,
        "normalized": span_text.lower(),
    }


def find_term_matches(text: str, label: str, terms: set[str]) -> list[dict[str, Any]]:
    spans: dict[tuple[str, int, int], dict[str, Any]] = {}
    for term in sorted(terms, key=len, reverse=True):
        for match in re.finditer(boundary_pattern(term), text, flags=re.IGNORECASE):
            add_match(spans, text, label, match.start(), match.end())
    return list(spans.values())


def split_role_candidates(value: str) -> set[str]:
    stripped_value = value.strip()
    candidates: set[str] = set()
    if stripped_value and not re.search(r"[|;,/]+", stripped_value):
        candidates.add(stripped_value)
    for part in re.split(r"[|;,/]+", value):
        stripped = part.strip()
        if len(stripped) >= 3:
            candidates.add(stripped)
    return {item for item in candidates if item}


def metadata_role_terms(row: dict[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for field in ("position", "primary_keyword"):
        value = str(row.get(field) or "").strip()
        if not value:
            continue
        for candidate in split_role_candidates(value):
            lowered = candidate.lower()
            if len(candidate.split()) > 5:
                continue
            if lowered in GENERIC_ROLE_TERMS:
                continue
            if candidate in ROLE_KEYWORDS:
                candidates.add(candidate)
                continue
            if any(token in lowered for token in ROLE_HINT_TOKENS):
                candidates.add(candidate)
    return candidates


def metadata_technology_terms(row: dict[str, Any]) -> set[str]:
    value = str(row.get("primary_keyword") or "").strip()
    candidates: set[str] = set()
    if value in TECHNOLOGY_TERMS:
        candidates.add(value)
    if value == "Golang":
        candidates.add("Golang")
    if value == ".NET":
        candidates.update({".NET", "ASP.NET", "C#"})
    return candidates


def find_regex_matches(text: str, label: str, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    spans: dict[tuple[str, int, int], dict[str, Any]] = {}
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_match(spans, text, label, match.start(), match.end())
    return list(spans.values())


def find_work_activity_matches(text: str) -> list[dict[str, Any]]:
    spans: dict[tuple[str, int, int], dict[str, Any]] = {}
    cursor = 0
    for line in text.splitlines():
        start = cursor
        end = cursor + len(line)
        cursor = end + 1
        stripped = line.strip()
        if len(stripped) < 18 or len(stripped) > 180:
            continue
        normalized = re.sub(r"^[\-\*\d\.\)\s]+", "", stripped).lower()
        if any(normalized.startswith(verb) for verb in WORK_ACTIVITY_VERBS):
            relative_start = line.index(stripped)
            add_match(spans, text, "WORK_ACTIVITY", start + relative_start, start + relative_start + len(stripped))
    return list(spans.values())


def prune_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generic_job_role_texts = {"lead", "support", "qa", "analyst", "architect", "manager", "developer", "engineer", "writer"}

    pruned: list[dict[str, Any]] = []
    job_roles = [entity for entity in entities if entity["label"] == "JOB_ROLE"]

    for entity in entities:
        normalized = entity["normalized"].strip().lower()
        if entity["label"] == "JOB_ROLE":
            if normalized in generic_job_role_texts:
                continue
            if ("/" in entity["text"] or "," in entity["text"] or len(entity["text"].split()) > 4) and any(
                other is not entity
                and other["start"] >= entity["start"]
                and other["end"] <= entity["end"]
                and len(other["text"]) < len(entity["text"])
                for other in job_roles
            ):
                continue
        if entity["label"] == "DEGREE" and re.search(r"\b(?:master|bachelor) of the\b", normalized):
            continue
        pruned.append(entity)

    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for entity in pruned:
        unique[(entity["label"], entity["start"], entity["end"])] = entity
    return sorted(unique.values(), key=lambda item: (item["start"], item["end"], item["label"]))


def extract_entities(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(row.get("text") or "")
    spans: dict[tuple[str, int, int], dict[str, Any]] = {}

    role_terms = ROLE_KEYWORDS | metadata_role_terms(row)
    technology_terms = TECHNOLOGY_TERMS | metadata_technology_terms(row)

    for entity in find_term_matches(text, "JOB_ROLE", role_terms):
        spans[(entity["label"], entity["start"], entity["end"])] = entity
    for entity in find_term_matches(text, "TECHNOLOGY", technology_terms):
        spans[(entity["label"], entity["start"], entity["end"])] = entity
    for entity in find_term_matches(text, "SKILL", SKILL_TERMS):
        spans[(entity["label"], entity["start"], entity["end"])] = entity
    for entity in find_term_matches(text, "INDUSTRY", INDUSTRY_TERMS):
        spans[(entity["label"], entity["start"], entity["end"])] = entity
    for entity in find_term_matches(text, "PROJECT_TYPE", PROJECT_TYPE_TERMS):
        spans[(entity["label"], entity["start"], entity["end"])] = entity
    for entity in find_regex_matches(text, "DEGREE", DEGREE_PATTERNS):
        spans[(entity["label"], entity["start"], entity["end"])] = entity
    for entity in find_regex_matches(text, "CERTIFICATION", CERTIFICATION_PATTERNS):
        spans[(entity["label"], entity["start"], entity["end"])] = entity
    for entity in find_work_activity_matches(text):
        spans[(entity["label"], entity["start"], entity["end"])] = entity

    return prune_entities(sorted(spans.values(), key=lambda item: (item["start"], item["end"], item["label"])))


def build_doccano_row(row: dict[str, Any], entities: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "doc_id": str(row.get("id") or ""),
        "text": str(row.get("text") or ""),
        "labels": [[entity["start"], entity["end"], entity["label"]] for entity in entities],
    }


def build_combined_row(row: dict[str, Any], entities: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_id": str(row.get("id") or ""),
        "document_type": str(row.get("doc_type") or "unknown"),
        "text": str(row.get("text") or ""),
        "entities": entities,
        "qualification_facts": [],
    }


def main() -> int:
    args = parse_args()
    for path in (args.cv_output, args.jd_output, args.combined_output, args.stats_output):
        ensure_writable(path, args.overwrite)

    stats = {
        "documents": 0,
        "by_type": Counter(),
        "label_counts": Counter(),
        "documents_with_entities": 0,
    }

    cv_rows: list[dict[str, Any]] = []
    jd_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []

    with args.input.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            doc_type = str(row.get("doc_type") or "").strip().lower()
            if not text or doc_type not in {"cv", "jd"}:
                continue

            entities = extract_entities(row)
            doccano_row = build_doccano_row(row, entities)
            combined_row = build_combined_row(row, entities)

            if doc_type == "cv":
                cv_rows.append(doccano_row)
            else:
                jd_rows.append(doccano_row)
            combined_rows.append(combined_row)

            stats["documents"] += 1
            stats["by_type"][doc_type] += 1
            if entities:
                stats["documents_with_entities"] += 1
            for entity in entities:
                stats["label_counts"][entity["label"]] += 1

    for path, rows in ((args.cv_output, cv_rows), (args.jd_output, jd_rows), (args.combined_output, combined_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats_payload = {
        "documents": stats["documents"],
        "documents_with_entities": stats["documents_with_entities"],
        "by_type": dict(stats["by_type"]),
        "label_counts": dict(stats["label_counts"]),
    }
    args.stats_output.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "cv_rows": len(cv_rows),
            "jd_rows": len(jd_rows),
            "combined_rows": len(combined_rows),
            "documents_with_entities": stats["documents_with_entities"],
            "label_counts": dict(stats["label_counts"]),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
