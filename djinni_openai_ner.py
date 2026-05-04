from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from djinni_gemini_ner import (
    DEFAULT_TIMEOUT_SECONDS,
    RESPONSE_JSON_SCHEMA,
    configure_logging,
    error_output_path,
    infer_doc_type,
    iter_rows,
    load_api_keys_from_file,
    load_processed_ids,
    match_field,
    maybe_save_raw_response,
    merge_api_keys,
    normalize_language,
    parse_text_specs,
    peek_fields,
    sanitize_annotation,
    split_text,
    stable_record_id,
    stringify,
    write_jsonl,
    DOC_TYPE_COLUMN_CANDIDATES,
    ID_COLUMN_CANDIDATES,
    LANGUAGE_COLUMN_CANDIDATES,
)


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
PROMPT_VERSION = "openai_stage1_v5"
RESPONSE_SCHEMA_NAME = "djinni_stage1_annotation"

SYSTEM_INSTRUCTION = """You are annotating Stage 1 training data for an academic thesis on IT CV-JD matching.

Your job is to read one chunk from a CV or job description and return precise structured labels that can later train a custom span-based RoBERTa NER model and support downstream O*NET mapping and hard-constraint scoring.

Follow these rules exactly:
1. Return JSON only, and make sure it matches the provided schema exactly.
2. Extract only information that is explicitly written in the chunk. Never invent missing skills, roles, or qualifications.
3. Every `text` span must be an exact substring copied from the chunk.
4. `start` and `end` offsets must use Python string indexing over the chunk only, with `start` inclusive and `end` exclusive.
5. Prefer the shortest contiguous span that preserves the meaning. Do not include extra punctuation, bullet markers, or surrounding filler words unless they are part of the real span.
6. Overlapping spans are allowed if both are genuinely valid.
7. If the chunk is not clearly English, still return JSON but set `document_language` accordingly.
8. If the chunk is not clearly about an IT candidate or IT job, set `is_it_document` to false and keep the arrays empty unless there are still obvious IT entities.
9. Use these labels:
   - TECHNOLOGY: programming languages, frameworks, libraries, cloud services, platforms, databases, operating systems, CI/CD tools, developer tools.
   - JOB_ROLE: job titles and role names such as Backend Engineer, Data Scientist, QA Engineer.
   - SKILL: only explicit non-technical competencies such as communication skills, stakeholder management, mentoring, negotiation, presentation skills.
   - WORK_ACTIVITY: only concrete work tasks or responsibilities. Keep specific verb-led or infinitive phrases like "build ETL pipelines", "design REST APIs", "mentor junior engineers", "to automate deployments".
   - INDUSTRY: only business sectors such as fintech, healthcare, e-commerce, telecom, logistics, retail.
   - PROJECT_TYPE: only specific project or product categories such as recommender systems, mobile games, CRM systems, web platforms, computer vision projects.
   - DEGREE: educational credentials, ideally including the field when it appears in the same contiguous span.
   - CERTIFICATION: named certifications only.
10. For `qualification_facts`, include only explicit years of experience, degree requirements/achievements, and certifications. For CVs, use operator `=` when the candidate states an achieved qualification or years of experience. For JDs, use the explicit comparator if present; otherwise use `unknown`.
11. Set `is_mandatory` to true only for hard requirements such as must, required, mandatory, minimum, at least, or equivalent wording.
12. Use lowercase canonical forms in `normalized` and `value` when possible, but do not change the original `text` span.
13. Never label English level indicators such as basic, intermediate, upper, advanced, fluent, or spans starting with "English Level" as SKILL.
14. Never label methodologies, technologies, or role-family phrases such as QA automation, Scrum, Agile, design patterns, OOP, data analysis, SQL, Python, Java, Android, or Data Analyst as SKILL unless the text explicitly presents them as a non-technical competency, which is rare.
15. For WORK_ACTIVITY, do not label noun phrases or discipline phrases such as testing, code review, bug fixing, software development, application development, manual testing, API testing, backend development, Android development, database design, or business analysis unless the span is an explicit action phrase led by a verb or infinitive.
16. For PROJECT_TYPE, do not label vague product-shape spans such as web application, mobile app, web development, pet project, web projects, published applications, desktop UI application, or bare generic nouns like application, app, apps, project, projects, platform, or product unless the span includes a specific modifier that makes it a concrete project type.
17. For INDUSTRY, do not label IT, security, cybersecurity, software, SaaS, data science, AI, analytics, engineering, product company, technology, game development, blockchain, networking, or computer vision as industries unless the span clearly denotes a business sector.
18. For DEGREE, do not label field-only spans like computer science or software engineering unless the credential word itself is included, such as degree, bachelor's, master's, PhD, diploma, BSc, MSc, or MBA.
19. For CERTIFICATION, keep only explicit named certifications. Do not label schools, training programs, courses, or generic vendor names as certifications.
20. When an explicit TECHNOLOGY or JOB_ROLE span is written, prefer to keep it even if the span is short.
21. If nothing relevant is present, return empty arrays.
"""

ENGLISH_LEVEL_TERMS = {
    "basic",
    "beginner",
    "elementary",
    "intermediate",
    "upper",
    "upper intermediate",
    "upper-intermediate",
    "pre",
    "pre intermediate",
    "pre-intermediate",
    "advanced",
    "fluent",
    "no english",
}

ENGLISH_LEVEL_CONTEXT_TERMS = {
    "conversational",
    "proficient",
    "proficiency",
}

NON_SKILL_TERMS = {
    "agile",
    "design pattern",
    "design patterns",
    "clean architecture",
    "hexagonal architecture",
    "layered architecture",
    "microservice architecture",
    "microservices architecture",
    "mvp",
    "mvvm",
    "object oriented programming",
    "object-oriented programming",
    "oop",
    "oop principles",
    "qa automation",
    "solid",
    "scrum",
}

NON_INDUSTRY_TERMS = {
    "it",
    "information technology",
    "security",
    "cybersecurity",
    "data science",
    "software",
    "software development",
    "saas",
    "analytics",
    "ai",
    "engineering",
    "product company",
    "technology",
    "game development",
    "blockchain",
    "computer vision",
    "gamedev",
    "networking",
}

GENERIC_PROJECT_TYPE_TERMS = {
    "application",
    "applications",
    "app",
    "apps",
    "project",
    "projects",
    "platform",
    "platforms",
    "product",
    "products",
    "mobile app",
    "mobile apps",
    "mobile application",
    "mobile applications",
    "web app",
    "web apps",
    "web application",
    "web applications",
    "web development",
    "backend project",
    "backend projects",
    "client server application",
    "client-server application",
    "desktop ui application",
    "desktop ui applications",
    "embedded project",
    "embedded projects",
    "finished project",
    "finished projects",
    "frontend project",
    "frontend projects",
    "pet project",
    "pet projects",
    "published application",
    "published applications",
    "real project",
    "real projects",
    "web project",
    "web projects",
}

GENERIC_WORK_ACTIVITY_TERMS = {
    "testing",
    "code review",
    "bug fixing",
    "fixing bugs",
    "manual testing",
    "api testing",
    "unit testing",
    "software development",
    "application development",
    "android development",
    "performance testing",
    "data visualization",
    "test automation",
    "refactoring",
    "backend development",
    "frontend development",
    "business analysis",
    "database design",
    "requirements analysis",
}

WORK_ACTIVITY_PREFIXES = (
    "to ",
    "responsible for ",
    "experience in ",
    "experience with ",
    "experienced in ",
    "experienced with ",
    "hands-on ",
    "hands on ",
    "involved in ",
    "focused on ",
)

NOISY_PREFIXES = (
    "strong ",
    "good ",
    "excellent ",
    "solid ",
    "deep ",
    "basic ",
    "experience in ",
    "experience with ",
    "experienced in ",
    "experienced with ",
    "understanding of ",
    "knowledge of ",
    "familiarity with ",
    "familiar with ",
    "hands-on ",
    "hands on ",
)

ACTION_VERB_TOKENS = {
    "analyze",
    "analyzed",
    "analyzing",
    "architect",
    "architected",
    "architecting",
    "automate",
    "automated",
    "automating",
    "build",
    "building",
    "built",
    "collaborate",
    "collaborated",
    "collaborating",
    "configure",
    "configured",
    "configuring",
    "coordinate",
    "coordinated",
    "coordinating",
    "create",
    "created",
    "creating",
    "debug",
    "debugged",
    "debugging",
    "deploy",
    "deployed",
    "deploying",
    "design",
    "designed",
    "designing",
    "develop",
    "developed",
    "developing",
    "document",
    "documented",
    "documenting",
    "estimate",
    "estimated",
    "estimating",
    "execute",
    "executed",
    "executing",
    "implement",
    "implemented",
    "implementing",
    "improve",
    "improved",
    "improving",
    "integrate",
    "integrated",
    "integrating",
    "lead",
    "leading",
    "led",
    "maintain",
    "maintained",
    "maintaining",
    "manage",
    "managed",
    "managing",
    "mentor",
    "mentored",
    "mentoring",
    "migrate",
    "migrated",
    "migrating",
    "monitor",
    "monitored",
    "monitoring",
    "optimize",
    "optimized",
    "optimizing",
    "own",
    "owned",
    "owning",
    "plan",
    "planned",
    "planning",
    "refactor",
    "refactored",
    "refactoring",
    "review",
    "reviewed",
    "reviewing",
    "support",
    "supported",
    "supporting",
    "test",
    "tested",
    "testing",
    "troubleshoot",
    "troubleshooting",
    "troubleshot",
    "write",
    "writing",
    "wrote",
    "written",
}

class OpenAIApiError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        body: str,
        *,
        retry_after_seconds: float | None = None,
        key_invalid: bool = False,
        quota_exhausted: bool = False,
        rate_limited: bool = False,
    ) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body
        self.retry_after_seconds = retry_after_seconds
        self.key_invalid = key_invalid
        self.quota_exhausted = quota_exhausted
        self.rate_limited = rate_limited


class AllOpenAIKeysExhaustedError(RuntimeError):
    pass


def parse_retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def build_openai_api_error(
    status_code: int,
    body: str,
    headers: dict[str, Any] | None = None,
) -> OpenAIApiError:
    retry_after_seconds = None
    if headers:
        retry_after_seconds = parse_retry_after_seconds(stringify(headers.get("Retry-After")))

    key_invalid = status_code in {401, 403}
    quota_exhausted = False
    rate_limited = status_code == 429

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        error_payload = payload.get("error") or {}
        error_type = stringify(error_payload.get("type")).lower()
        error_code = stringify(error_payload.get("code")).lower()
        message = stringify(error_payload.get("message")).lower()

        key_invalid = key_invalid or error_type in {"authentication_error", "invalid_api_key"}
        quota_exhausted = error_code == "insufficient_quota" or error_type == "insufficient_quota"
        rate_limited = rate_limited or error_code == "rate_limit_exceeded"

        if not quota_exhausted:
            quota_exhausted = (
                "insufficient_quota" in message
                or "billing" in message
                or "check your plan" in message
                or "please check your plan" in message
            )

    return OpenAIApiError(
        status_code,
        body,
        retry_after_seconds=retry_after_seconds,
        key_invalid=key_invalid,
        quota_exhausted=quota_exhausted,
        rate_limited=rate_limited,
    )


def build_prompt(record_id: str, doc_type: str, chunk_index: int, chunk_start: int, chunk_text: str) -> str:
    return f"""Annotate this recruitment document chunk for Stage 1 of the thesis pipeline.

record_id: {record_id}
document_type: {doc_type}
chunk_index: {chunk_index}
chunk_start_in_full_document: {chunk_start}

Goal:
- produce weak NER labels for IT CV/JD matching
- keep exact spans for later span-based RoBERTa training
- keep explicit qualification facts for later hard-constraint scoring

Output requirements:
- return JSON only
- offsets must be relative to the chunk below, not the full document
- do not add markdown or explanations

Critical label exclusions:
- do not label English levels such as basic, intermediate, upper, advanced, fluent, or phrases like upper-intermediate English as SKILL
- keep SKILL only for explicit non-technical competencies
- do not label methodologies or technical phrases like QA automation, Agile, Scrum, design patterns, OOP, object-oriented programming, or data analysis as SKILL
- keep WORK_ACTIVITY only for specific action phrases, ideally verb-led or infinitive-led
- do not label noun phrases like backend development, database design, business analysis, software development, code review, bug fixing, or testing as WORK_ACTIVITY unless they are inside a more explicit action phrase
- do not label vague spans like web application, mobile app, web development, pet project, published applications, desktop ui application, or bare generic spans like application, app, project, platform, or product as PROJECT_TYPE
- keep INDUSTRY only for clear business sectors such as healthcare, fintech, e-commerce, telecom, not product company, technology, game development, blockchain, networking, or computer vision
- do not label field-only spans like computer science or software engineering as DEGREE unless the credential word is part of the span
- keep CERTIFICATION only for explicit named certifications, not schools, courses, or training programs
- be permissive for explicit TECHNOLOGY and JOB_ROLE spans

Chunk text:
<<<TEXT
{chunk_text}
TEXT
"""


def normalize_annotation_text(value: str) -> str:
    return re.sub(r"\s+", " ", stringify(value)).strip().lower()


def is_english_level_text(value: str) -> bool:
    normalized = normalize_annotation_text(value)
    if normalized in ENGLISH_LEVEL_TERMS:
        return True
    if normalized.startswith("english level"):
        return True
    if normalized.startswith("level of english"):
        return True
    if "english" in normalized and any(term in normalized for term in ENGLISH_LEVEL_TERMS | ENGLISH_LEVEL_CONTEXT_TERMS):
        return True
    return False


def strip_activity_prefix(value: str) -> str:
    candidate = value
    changed = True
    while changed:
        changed = False
        for prefix in WORK_ACTIVITY_PREFIXES:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix):].strip()
                changed = True
    return candidate


def has_verb_led_activity(value: str) -> bool:
    candidate = strip_activity_prefix(value)
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", candidate)
    if len(tokens) < 2:
        return False
    return tokens[0] in ACTION_VERB_TOKENS


def is_non_skill_text(value: str) -> bool:
    normalized = normalize_annotation_text(value)
    if not normalized:
        return False
    if normalized in NON_SKILL_TERMS:
        return True
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"\bqa automation\b",
            r"\bagile\b",
            r"\bscrum\b",
            r"\bdesign patterns?\b",
            r"\boop\b",
            r"\bobject[- ]oriented programming\b",
            r"\bmvvm\b",
            r"\bmvp\b",
            r"\bsolid\b",
            r"\bclean architecture\b",
            r"\bhexagonal architecture\b",
            r"\blayered architecture\b",
            r"\bmicroservices? architecture\b",
        )
    )


DEGREE_KEYWORD_PATTERNS = (
    r"\bbachelor(?:'s)?\b",
    r"\bmaster(?:'s)?\b",
    r"\bph\.?d\b",
    r"\bdoctorate\b",
    r"\bdegree\b",
    r"\bdiploma\b",
    r"\bb\.?\s?sc\b",
    r"\bm\.?\s?sc\b",
    r"\bmba\b",
)

DEGREE_NOISE_PATTERNS = (
    r"\bbachelor thesis\b",
    r"\bmaster thesis\b",
    r"\bph\.?d student\b",
    r"\bcomputer science background\b",
    r"\bsoftware engineering background\b",
)

CERTIFICATION_GENERIC_TERMS = {
    "aws",
    "certificate",
    "certification",
    "certifications",
    "microsoft certifications",
}

CERTIFICATION_NOISE_PATTERNS = (
    r"\bcourse\b",
    r"\bschool\b",
    r"\btraining\b",
    r"\btraining program\b",
    r"\bsubject\b",
    r"\bhatchery\b",
    r"\bdiploma projects?\b",
)

CERTIFICATION_PATTERNS = (
    r"\baws\b.*\b(certified|associate|professional|specialty)\b",
    r"\bazure\b.*\b(administrator|fundamentals|solutions architect|developer|associate|expert)\b",
    r"\bgoogle cloud\b.*\b(professional|associate)\b",
    r"\boracle certified\b",
    r"\bsitecore certified\b",
    r"\bcertified kubernetes\b",
    r"\bistqb\b",
    r"\bitil\b",
    r"\bpmp\b",
    r"\bccna\b",
    r"\brhcsa\b",
    r"\brhce\b",
    r"\bcka\b",
    r"\bckad\b",
)


def is_valid_degree_text(value: str) -> bool:
    normalized = normalize_annotation_text(value)
    if not normalized:
        return False
    if any(re.search(pattern, normalized) for pattern in DEGREE_NOISE_PATTERNS):
        return False
    return any(re.search(pattern, normalized) for pattern in DEGREE_KEYWORD_PATTERNS)


def is_valid_certification_text(value: str) -> bool:
    normalized = normalize_annotation_text(value)
    if not normalized:
        return False
    if normalized in CERTIFICATION_GENERIC_TERMS:
        return False
    if any(re.search(pattern, normalized) for pattern in CERTIFICATION_NOISE_PATTERNS):
        return False
    if any(re.search(pattern, normalized) for pattern in CERTIFICATION_PATTERNS):
        return True
    if "certified" in normalized:
        return True
    if ("certificate" in normalized or "certification" in normalized) and len(normalized.split()) > 1:
        return True
    return False


def should_drop_entity(entity: dict[str, Any]) -> bool:
    label = stringify(entity.get("label")).upper()
    normalized = normalize_annotation_text(entity.get("normalized") or entity.get("text"))

    if not label or not normalized:
        return False

    if label == "ABILITY":
        return True

    if label == "SKILL" and is_english_level_text(normalized):
        return True

    if label == "SKILL" and is_non_skill_text(normalized):
        return True

    if label == "INDUSTRY" and normalized in NON_INDUSTRY_TERMS:
        return True

    if label == "PROJECT_TYPE" and normalized in GENERIC_PROJECT_TYPE_TERMS:
        return True

    if label == "DEGREE" and not is_valid_degree_text(normalized):
        return True

    if label == "CERTIFICATION" and not is_valid_certification_text(normalized):
        return True

    if label == "WORK_ACTIVITY":
        if normalized in GENERIC_WORK_ACTIVITY_TERMS:
            return True
        if len(normalized.split()) == 1:
            return True
        if not has_verb_led_activity(normalized):
            return True

    return False


def trim_entity_span(entity: dict[str, Any]) -> None:
    text = stringify(entity.get("text"))
    if not text:
        return
        
    start = entity.get("start")
    if not isinstance(start, int):
        return
        
    changed = True
    while changed:
        changed = False
        lower_text = text.lower()
        for prefix in NOISY_PREFIXES:
            if lower_text.startswith(prefix):
                prefix_len = len(prefix)
                while prefix_len < len(text) and text[prefix_len].isspace():
                    prefix_len += 1
                text = text[prefix_len:]
                start += prefix_len
                changed = True
                break
                
    entity["text"] = text
    if entity.get("normalized"):
        entity["normalized"] = normalize_annotation_text(text)
    entity["start"] = start


def refine_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    refined = dict(annotation)
    kept_entities: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()

    for entity in annotation.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        trim_entity_span(entity)
        if should_drop_entity(entity):
            continue
        try:
            start = int(entity.get("start", -1))
            end = int(entity.get("end", -1))
        except (TypeError, ValueError):
            start, end = -1, -1
        key = (
            stringify(entity.get("label")).upper(),
            start,
            end,
            stringify(entity.get("text")),
        )
        if key in seen:
            continue
        seen.add(key)
        kept_entities.append(entity)

    refined["entities"] = kept_entities
    return refined


def extract_message_text(raw: dict[str, Any]) -> str:
    choices = raw.get("choices") or []
    if not choices:
        raise ValueError(f"No choices returned by OpenAI: {json.dumps(raw)[:500]}")

    choice = choices[0]
    finish_reason = stringify(choice.get("finish_reason")).lower()
    if finish_reason and finish_reason not in {"stop"}:
        raise ValueError(f"OpenAI finish_reason={finish_reason}: {json.dumps(choice)[:500]}")

    message = choice.get("message") or {}
    refusal = stringify(message.get("refusal"))
    if refusal:
        raise ValueError(f"OpenAI refusal: {refusal}")

    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append(stringify(part.get("text")))
        text = "".join(parts).strip()
    else:
        text = ""

    if not text:
        raise ValueError(f"OpenAI returned empty content: {json.dumps(raw)[:500]}")
    return text


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")
    return payload


class OpenAIChatCompletionsClient:
    def __init__(
        self,
        api_keys: list[str],
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 5,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        if not api_keys:
            raise ValueError("At least one OpenAI API key is required.")
        self.api_keys = api_keys
        self.active_key_index = 0
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def generate_annotation(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": RESPONSE_SCHEMA_NAME,
                    "strict": True,
                    "schema": RESPONSE_JSON_SCHEMA,
                },
            },
        }

        encoded = json.dumps(payload).encode("utf-8")
        while True:
            headers = {
                "Authorization": f"Bearer {self.api_keys[self.active_key_index]}",
                "Content-Type": "application/json",
            }

            last_error: Exception | None = None
            rotated_key = False

            for attempt in range(1, self.max_retries + 1):
                req = request.Request(self.endpoint, data=encoded, headers=headers, method="POST")
                try:
                    with request.urlopen(req, timeout=self.timeout_seconds) as response:
                        raw_body = response.read().decode("utf-8")
                    raw = json.loads(raw_body)
                    text = extract_message_text(raw)
                    return parse_model_json(text), raw
                except error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    api_error = build_openai_api_error(exc.code, body, dict(exc.headers.items()))
                    last_error = api_error

                    if api_error.key_invalid or api_error.quota_exhausted:
                        if self.rotate_api_key(api_error):
                            rotated_key = True
                            break
                        if len(self.api_keys) == 1 or self.active_key_index + 1 >= len(self.api_keys):
                            raise AllOpenAIKeysExhaustedError(
                                "All configured OpenAI API keys are unavailable. Add a new key and rerun with --resume."
                            ) from api_error

                    retryable_status = api_error.status_code in {408, 409, 429} or api_error.status_code >= 500
                    if not retryable_status and not api_error.rate_limited:
                        break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc

                if attempt < self.max_retries:
                    if isinstance(last_error, OpenAIApiError) and last_error.retry_after_seconds is not None:
                        sleep_seconds = last_error.retry_after_seconds
                    else:
                        sleep_seconds = self.retry_backoff_seconds * attempt
                    logging.warning(
                        "OpenAI call failed on attempt %s/%s; retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)

            if rotated_key:
                continue

            assert last_error is not None
            raise last_error

    def rotate_api_key(self, reason: OpenAIApiError) -> bool:
        if self.active_key_index + 1 >= len(self.api_keys):
            return False
        previous_slot = self.active_key_index + 1
        self.active_key_index += 1
        next_slot = self.active_key_index + 1
        logging.warning(
            "OpenAI API key slot %s is unavailable (%s). Switching to key slot %s/%s.",
            previous_slot,
            "quota exhausted" if reason.quota_exhausted else "invalid key",
            next_slot,
            len(self.api_keys),
        )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Djinni CV/JD text with OpenAI zero-shot NER spans for downstream span-based NER training."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV or JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file for annotations.")
    parser.add_argument("--api-key", help="Single OpenAI API key. Defaults to OPENAI_API_KEY if set.")
    parser.add_argument(
        "--api-keys-file",
        type=Path,
        help="Text file with one OpenAI API key per line. Keys are rotated on quota/auth failures.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"HTTP endpoint. Default: {DEFAULT_ENDPOINT}")
    parser.add_argument("--id-column", help="Column used as the stable source identifier.")
    parser.add_argument("--text-column", help="Single text column to annotate.")
    parser.add_argument(
        "--text-columns",
        help="Comma-separated text specs. Example: cv:resume_text,jd:job_description. If set, overrides --text-column.",
    )
    parser.add_argument(
        "--document-type",
        choices=("auto", "cv", "jd"),
        default="auto",
        help="Fallback document type when it cannot be inferred from metadata.",
    )
    parser.add_argument("--doc-type-column", help="Column storing document type/source metadata.")
    parser.add_argument("--language-column", help="Column storing language metadata.")
    parser.add_argument("--max-records", type=int, help="Stop after N source rows.")
    parser.add_argument("--max-documents", type=int, help="Stop after N emitted document annotations.")
    parser.add_argument("--max-chars-per-call", type=int, default=12000, help="Chunk long documents before calling OpenAI.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap in characters between long text chunks.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between OpenAI requests.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout per request.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per OpenAI call.")
    parser.add_argument("--require-english", action="store_true", help="Skip outputs whose inferred language is not English.")
    parser.add_argument("--require-it", action="store_true", help="Skip outputs whose inferred domain is not IT.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output JSONL by skipping seen ids.")
    parser.add_argument("--raw-output-dir", type=Path, help="Optional directory for raw OpenAI responses.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def load_api_keys(args: argparse.Namespace) -> list[str]:
    keys: list[str] = []

    env_multi = os.environ.get("OPENAI_API_KEYS")
    if env_multi:
        keys.extend(re.split(r"[\r\n,]+", env_multi))

    env_single = os.environ.get("OPENAI_API_KEY")
    if env_single:
        keys.append(env_single)

    if args.api_keys_file:
        keys.extend(load_api_keys_from_file(args.api_keys_file))

    if args.api_key:
        keys.append(args.api_key)

    return merge_api_keys(keys)


def annotate_document(
    client: OpenAIChatCompletionsClient,
    record_id: str,
    doc_type: str,
    text: str,
    max_chars_per_call: int,
    chunk_overlap: int,
    sleep_seconds: float,
    raw_output_dir: Path | None,
) -> dict[str, Any]:
    chunks = split_text(text, max_chars_per_call, chunk_overlap)
    chunk_outputs: list[dict[str, Any]] = []

    for chunk_index, (chunk_start, chunk_text) in enumerate(chunks):
        prompt = build_prompt(record_id, doc_type, chunk_index, chunk_start, chunk_text)
        annotation, raw = client.generate_annotation(prompt)
        chunk_outputs.append(annotation)
        maybe_save_raw_response(raw_output_dir, record_id, chunk_index, raw)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return refine_annotation(sanitize_annotation(text, chunks, chunk_outputs))


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    api_keys = load_api_keys(args)
    if not api_keys:
        raise SystemExit("OpenAI API key missing. Pass --api-key, --api-keys-file, or set OPENAI_API_KEY / OPENAI_API_KEYS.")
    logging.info("Loaded %s OpenAI API key(s).", len(api_keys))

    fields = peek_fields(args.input)
    if not fields:
        raise SystemExit(f"No rows found in {args.input}.")

    id_column = match_field(fields, args.id_column, ID_COLUMN_CANDIDATES)
    doc_type_column = match_field(fields, args.doc_type_column, DOC_TYPE_COLUMN_CANDIDATES)
    language_column = match_field(fields, args.language_column, LANGUAGE_COLUMN_CANDIDATES)
    text_specs = parse_text_specs(fields, args)

    if args.resume:
        processed_ids = load_processed_ids(args.output)
        logging.info("Loaded %s processed ids from %s", len(processed_ids), args.output)
    else:
        processed_ids = set()

    client = OpenAIChatCompletionsClient(
        api_keys=api_keys,
        model=args.model,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    emitted_documents = 0
    visited_rows = 0

    for row_index, row in enumerate(iter_rows(args.input), start=1):
        visited_rows += 1
        if args.max_records and visited_rows > args.max_records:
            break

        row_doc_type_value = stringify(row.get(doc_type_column)) if doc_type_column else ""
        row_language_value = stringify(row.get(language_column)) if language_column else ""

        for text_spec in text_specs:
            text_value = stringify(row.get(text_spec.column))
            if not text_value:
                continue

            record_id = stable_record_id(row, id_column, row_index, text_value, text_spec.column)
            if record_id in processed_ids:
                continue

            doc_type = infer_doc_type(row_doc_type_value, text_spec.doc_type, text_spec.column)
            source_language = normalize_language(row_language_value)
            logging.info("Annotating %s (%s, %s chars)", record_id, doc_type, len(text_value))

            try:
                annotation = annotate_document(
                    client=client,
                    record_id=record_id,
                    doc_type=doc_type,
                    text=text_value,
                    max_chars_per_call=args.max_chars_per_call,
                    chunk_overlap=args.chunk_overlap,
                    sleep_seconds=args.sleep_seconds,
                    raw_output_dir=args.raw_output_dir,
                )
            except AllOpenAIKeysExhaustedError as exc:
                write_jsonl(
                    error_output_path(args.output),
                    {
                        "record_id": record_id,
                        "row_index": row_index,
                        "source_column": text_spec.column,
                        "error": str(exc),
                        "fatal": True,
                    },
                )
                logging.error("%s", exc)
                logging.error(
                    "Progress is saved in %s. Add a new OpenAI key and rerun the same command with --resume.",
                    args.output,
                )
                return 2
            except Exception as exc:  # noqa: BLE001
                write_jsonl(
                    error_output_path(args.output),
                    {
                        "record_id": record_id,
                        "row_index": row_index,
                        "source_column": text_spec.column,
                        "error": str(exc),
                    },
                )
                logging.exception("Failed to annotate %s", record_id)
                continue

            if args.require_english and annotation["document_language"] != "en":
                logging.info("Skipping %s because inferred language is %s", record_id, annotation["document_language"])
                continue
            if args.require_it and not annotation["is_it_document"]:
                logging.info("Skipping %s because the document is not classified as IT", record_id)
                continue

            output_row = {
                "record_id": record_id,
                "source_path": str(args.input),
                "source_row_index": row_index,
                "source_column": text_spec.column,
                "source_language": source_language,
                "document_type": doc_type,
                "provider": "openai",
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
                "text": text_value,
                "document_language": annotation["document_language"],
                "is_it_document": annotation["is_it_document"],
                "summary": annotation["summary"],
                "entities": annotation["entities"],
                "qualification_facts": annotation["qualification_facts"],
                "text_length": annotation["text_length"],
            }
            write_jsonl(args.output, output_row)
            processed_ids.add(record_id)
            emitted_documents += 1

            if args.max_documents and emitted_documents >= args.max_documents:
                logging.info("Reached --max-documents=%s", args.max_documents)
                return 0

    logging.info("Finished. Visited %s source rows and emitted %s documents.", visited_rows, emitted_documents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
