from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


FILE_ALIASES: dict[str, tuple[str, ...]] = {
    "occupation_data": ("occupation data", "occupation_data", "occupationdata"),
    "alternate_titles": ("alternate titles", "alternate_titles", "alternatetitles"),
    "skills": ("skills",),
    "knowledge": ("knowledge",),
    "abilities": ("abilities",),
    "work_activities": ("work activities", "work_activities", "workactivities"),
    "technology_skills": ("technology skills", "technology_skills", "technologyskills"),
    "task_statements": ("task statements", "task_statements", "taskstatements"),
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

SHORT_KEEP_TOKENS = {"ai", "bi", "c", "c#", "c++", "go", "js", "ml", "qa", "ui", "ux"}

LABEL_TARGET_TYPES: dict[str, tuple[str, ...]] = {
    "JOB_ROLE": ("occupation", "alternate_title"),
    "SKILL": ("skill", "knowledge", "ability"),
    "WORK_ACTIVITY": ("work_activity", "task_statement"),
    "TECHNOLOGY": ("technology_skill", "skill", "knowledge"),
    "PROJECT_TYPE": ("task_statement", "work_activity", "technology_skill"),
}


@dataclass(frozen=True)
class OnetEntry:
    entry_id: str
    entry_type: str
    title: str
    normalized_text: str
    aliases: tuple[str, ...]
    alias_normalized: tuple[str, ...]
    onetsoc_code: str
    occupation_title: str
    source: str
    element_id: str = ""
    importance: float | None = None
    level: float | None = None
    commodity_code: str = ""
    commodity_title: str = ""
    hot_technology: bool = False
    in_demand: bool = False


@dataclass
class OnetMatch:
    entry: OnetEntry
    score: float
    lexical_score: float
    semantic_score: float | None
    context_boost: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.entry)
        payload.update(
            {
                "score": self.score,
                "lexical_score": self.lexical_score,
                "semantic_score": self.semantic_score,
                "context_boost": self.context_boost,
            }
        )
        return payload


def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def canonicalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def canonicalize_filename(value: str) -> str:
    return canonicalize_header(Path(value).stem)


def normalize_text(value: str) -> str:
    lowered = stringify(value).strip().lower()
    lowered = lowered.replace("’", "'")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def tokenize(value: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9][a-z0-9.+#/\-]*", normalize_text(value))
    filtered: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) == 1 and token not in SHORT_KEEP_TOKENS:
            continue
        filtered.append(token)
    return tuple(filtered)


def character_ngrams(value: str, size: int = 3) -> set[str]:
    collapsed = re.sub(r"\s+", "", normalize_text(value))
    if len(collapsed) < size:
        return {collapsed} if collapsed else set()
    return {collapsed[index : index + size] for index in range(len(collapsed) - size + 1)}


def safe_float(value: Any) -> float | None:
    raw = stringify(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def sniff_delimiter(sample: str, fallback: str = "\t") -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
    except csv.Error:
        return fallback
    return dialect.delimiter


def load_tabular_rows(path: Path) -> list[dict[str, str]]:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    delimiter = "\t" if path.suffix.lower() == ".txt" else ","
    delimiter = sniff_delimiter(sample, fallback=delimiter)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows: list[dict[str, str]] = []
        for row in reader:
            normalized_row = {
                canonicalize_header(stringify(key)): stringify(value).strip()
                for key, value in row.items()
                if key is not None
            }
            if normalized_row:
                rows.append(normalized_row)
    return rows


def first_value(row: dict[str, str], *candidates: str) -> str:
    for candidate in candidates:
        key = canonicalize_header(candidate)
        if key in row and row[key]:
            return row[key]
    return ""


def resolve_onet_files(onet_dir: Path) -> dict[str, Path]:
    if not onet_dir.exists():
        raise SystemExit(f"O*NET directory not found: {onet_dir}")
    files = {canonicalize_filename(path.name): path for path in onet_dir.iterdir() if path.is_file()}
    resolved: dict[str, Path] = {}
    for logical_name, aliases in FILE_ALIASES.items():
        for alias in aliases:
            if canonicalize_header(alias) in files:
                resolved[logical_name] = files[canonicalize_header(alias)]
                break
    if "occupation_data" not in resolved:
        raise SystemExit(
            "O*NET folder must contain Occupation Data file (for example 'Occupation Data.txt')."
        )
    return resolved


def occupation_lookup(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        code = first_value(row, "O*NET-SOC Code", "onetsoc_code")
        title = first_value(row, "Title", "title")
        description = first_value(row, "Description", "description")
        if not code or not title:
            continue
        lookup[code] = {"title": title, "description": description}
    return lookup


def build_descriptor_entries(
    rows: Iterable[dict[str, str]],
    *,
    entry_type: str,
    source_name: str,
    occupations: dict[str, dict[str, str]],
) -> list[OnetEntry]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        code = first_value(row, "O*NET-SOC Code", "onetsoc_code")
        element_name = first_value(row, "Element Name", "element_name", "Title", "title")
        if not code or not element_name:
            continue
        element_id = first_value(row, "Element ID", "element_id")
        scale_id = first_value(row, "Scale ID", "scale_id").upper()
        data_value = safe_float(first_value(row, "Data Value", "data_value"))
        key = (code, element_id or element_name, element_name)
        entry = grouped.setdefault(
            key,
            {
                "code": code,
                "element_id": element_id,
                "title": element_name,
                "importance": None,
                "level": None,
            },
        )
        if data_value is None:
            continue
        if scale_id == "IM":
            entry["importance"] = data_value
        elif scale_id == "LV":
            entry["level"] = data_value

    entries: list[OnetEntry] = []
    for payload in grouped.values():
        occupation = occupations.get(payload["code"], {})
        title = payload["title"]
        entries.append(
            OnetEntry(
                entry_id=f"{entry_type}::{payload['code']}::{payload['element_id'] or normalize_text(title)}",
                entry_type=entry_type,
                title=title,
                normalized_text=normalize_text(title),
                aliases=(),
                alias_normalized=(),
                onetsoc_code=payload["code"],
                occupation_title=occupation.get("title", ""),
                source=source_name,
                element_id=payload["element_id"] or "",
                importance=payload["importance"],
                level=payload["level"],
            )
        )
    return entries


def build_technology_entries(
    rows: Iterable[dict[str, str]],
    *,
    occupations: dict[str, dict[str, str]],
) -> list[OnetEntry]:
    entries: dict[tuple[str, str], OnetEntry] = {}
    for row in rows:
        code = first_value(row, "O*NET-SOC Code", "onetsoc_code")
        example = first_value(row, "Example", "example")
        if not code or not example:
            continue
        occupation = occupations.get(code, {})
        commodity_code = first_value(row, "Commodity Code", "commodity_code")
        commodity_title = first_value(row, "Commodity Title", "commodity_title")
        key = (code, normalize_text(example))
        entries[key] = OnetEntry(
            entry_id=f"technology_skill::{code}::{normalize_text(example)}",
            entry_type="technology_skill",
            title=example,
            normalized_text=normalize_text(example),
            aliases=tuple(alias for alias in (commodity_title,) if alias),
            alias_normalized=tuple(normalize_text(alias) for alias in (commodity_title,) if alias),
            onetsoc_code=code,
            occupation_title=occupation.get("title", ""),
            source="Technology Skills",
            commodity_code=commodity_code,
            commodity_title=commodity_title,
            hot_technology=first_value(row, "Hot Technology", "hot_technology").upper() == "Y",
            in_demand=first_value(row, "In Demand", "in_demand").upper() == "Y",
        )
    return list(entries.values())


def build_task_entries(
    rows: Iterable[dict[str, str]],
    *,
    occupations: dict[str, dict[str, str]],
) -> list[OnetEntry]:
    entries: list[OnetEntry] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        code = first_value(row, "O*NET-SOC Code", "onetsoc_code")
        task = first_value(row, "Task", "task", "Task Statement", "task_statement")
        task_id = first_value(row, "Task ID", "task_id")
        if not code or not task:
            continue
        key = (code, normalize_text(task))
        if key in seen:
            continue
        seen.add(key)
        occupation = occupations.get(code, {})
        entries.append(
            OnetEntry(
                entry_id=f"task_statement::{code}::{task_id or normalize_text(task)}",
                entry_type="task_statement",
                title=task,
                normalized_text=normalize_text(task),
                aliases=(),
                alias_normalized=(),
                onetsoc_code=code,
                occupation_title=occupation.get("title", ""),
                source="Task Statements",
                element_id=task_id,
            )
        )
    return entries


def build_role_entries(
    occupations: dict[str, dict[str, str]],
    alternate_title_rows: Iterable[dict[str, str]] | None,
) -> list[OnetEntry]:
    entries: list[OnetEntry] = []
    for code, payload in occupations.items():
        title = payload.get("title", "")
        description = payload.get("description", "")
        aliases = tuple(alias for alias in (description,) if alias)
        entries.append(
            OnetEntry(
                entry_id=f"occupation::{code}",
                entry_type="occupation",
                title=title,
                normalized_text=normalize_text(title),
                aliases=aliases,
                alias_normalized=tuple(normalize_text(alias) for alias in aliases),
                onetsoc_code=code,
                occupation_title=title,
                source="Occupation Data",
            )
        )

    if alternate_title_rows is None:
        return entries

    seen: set[tuple[str, str]] = set()
    for row in alternate_title_rows:
        code = first_value(row, "O*NET-SOC Code", "onetsoc_code")
        title = first_value(row, "Alternate Title", "alternate_title", "Title", "title")
        if not code or not title or code not in occupations:
            continue
        key = (code, normalize_text(title))
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            OnetEntry(
                entry_id=f"alternate_title::{code}::{normalize_text(title)}",
                entry_type="alternate_title",
                title=title,
                normalized_text=normalize_text(title),
                aliases=(),
                alias_normalized=(),
                onetsoc_code=code,
                occupation_title=occupations[code]["title"],
                source="Alternate Titles",
            )
        )
    return entries


def prepare_onet_index(onet_dir: Path) -> tuple[list[OnetEntry], dict[str, Any]]:
    resolved = resolve_onet_files(onet_dir)
    occupation_rows = load_tabular_rows(resolved["occupation_data"])
    occupations = occupation_lookup(occupation_rows)
    alternate_title_rows = (
        load_tabular_rows(resolved["alternate_titles"]) if "alternate_titles" in resolved else None
    )

    entries: list[OnetEntry] = []
    entries.extend(build_role_entries(occupations, alternate_title_rows))

    descriptor_specs = (
        ("skills", "skill", "Skills"),
        ("knowledge", "knowledge", "Knowledge"),
        ("abilities", "ability", "Abilities"),
        ("work_activities", "work_activity", "Work Activities"),
    )
    for file_key, entry_type, source_name in descriptor_specs:
        if file_key not in resolved:
            continue
        entries.extend(
            build_descriptor_entries(
                load_tabular_rows(resolved[file_key]),
                entry_type=entry_type,
                source_name=source_name,
                occupations=occupations,
            )
        )

    if "technology_skills" in resolved:
        entries.extend(build_technology_entries(load_tabular_rows(resolved["technology_skills"]), occupations=occupations))

    if "task_statements" in resolved:
        entries.extend(build_task_entries(load_tabular_rows(resolved["task_statements"]), occupations=occupations))

    summary = {
        "source_dir": str(onet_dir),
        "resolved_files": {key: str(path) for key, path in resolved.items()},
        "occupations": len(occupations),
        "entries": len(entries),
        "entry_types": {
            entry_type: sum(1 for entry in entries if entry.entry_type == entry_type)
            for entry_type in sorted({entry.entry_type for entry in entries})
        },
    }
    return entries, summary


def write_onet_index(path: Path, entries: Iterable[OnetEntry], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def load_onet_index(path: Path) -> list[OnetEntry]:
    if not path.exists():
        raise SystemExit(f"O*NET index not found: {path}")
    entries: list[OnetEntry] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            entries.append(
                OnetEntry(
                    entry_id=payload["entry_id"],
                    entry_type=payload["entry_type"],
                    title=payload["title"],
                    normalized_text=payload["normalized_text"],
                    aliases=tuple(payload.get("aliases") or []),
                    alias_normalized=tuple(payload.get("alias_normalized") or []),
                    onetsoc_code=payload["onetsoc_code"],
                    occupation_title=payload.get("occupation_title", ""),
                    source=payload.get("source", ""),
                    element_id=payload.get("element_id", ""),
                    importance=payload.get("importance"),
                    level=payload.get("level"),
                    commodity_code=payload.get("commodity_code", ""),
                    commodity_title=payload.get("commodity_title", ""),
                    hot_technology=bool(payload.get("hot_technology", False)),
                    in_demand=bool(payload.get("in_demand", False)),
                )
            )
    return entries


class TransformerEmbedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 1), dtype=torch.float32)
        batches: list[torch.Tensor] = []
        with torch.no_grad():
            for index in range(0, len(texts), self.batch_size):
                batch_texts = texts[index : index + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batches.append(F.normalize(pooled, dim=1).cpu())
        return torch.cat(batches, dim=0)


class OnetMapper:
    def __init__(
        self,
        entries: list[OnetEntry],
        *,
        embedding_model: str | None = None,
        embedding_device: str | None = None,
        lexical_candidate_limit: int = 256,
        min_token_overlap: int = 1,
    ) -> None:
        self.entries = entries
        self.lexical_candidate_limit = lexical_candidate_limit
        self.min_token_overlap = min_token_overlap
        self.entry_types = {entry.entry_type for entry in entries}
        self.tokens_by_index = [set(tokenize(entry.title + " " + " ".join(entry.aliases))) for entry in entries]
        self.trigrams_by_index = [character_ngrams(entry.title + " " + " ".join(entry.aliases)) for entry in entries]
        self.token_to_indices: dict[str, set[int]] = {}
        for index, tokens in enumerate(self.tokens_by_index):
            for token in tokens:
                self.token_to_indices.setdefault(token, set()).add(index)
        self.embedder = TransformerEmbedder(embedding_model, device=embedding_device) if embedding_model else None
        self.entry_embeddings: torch.Tensor | None = None
        if self.embedder is not None:
            logging.info("Encoding %s O*NET entries with %s", len(entries), embedding_model)
            self.entry_embeddings = self.embedder.encode(
                [entry.title if not entry.aliases else f"{entry.title}. {'; '.join(entry.aliases)}" for entry in entries]
            )

    def supported_targets(self, label: str) -> tuple[str, ...]:
        return LABEL_TARGET_TYPES.get(label.upper(), ())

    def candidate_indices(self, text: str, allowed_types: set[str]) -> list[int]:
        query_tokens = set(tokenize(text))
        matched_indices: set[int] = set()
        overlap_scores: dict[int, float] = {}

        for token in query_tokens:
            for index in self.token_to_indices.get(token, ()):
                if self.entries[index].entry_type not in allowed_types:
                    continue
                overlap = len(query_tokens & self.tokens_by_index[index])
                if overlap < self.min_token_overlap:
                    continue
                matched_indices.add(index)
                union = max(len(query_tokens | self.tokens_by_index[index]), 1)
                overlap_scores[index] = max(overlap_scores.get(index, 0.0), overlap / union)

        if not matched_indices:
            query_trigrams = character_ngrams(text)
            for index, entry_trigrams in enumerate(self.trigrams_by_index):
                if self.entries[index].entry_type not in allowed_types:
                    continue
                if not query_trigrams or not entry_trigrams:
                    continue
                overlap = len(query_trigrams & entry_trigrams)
                if overlap == 0:
                    continue
                union = max(len(query_trigrams | entry_trigrams), 1)
                overlap_scores[index] = overlap / union
                matched_indices.add(index)

        ranked = sorted(matched_indices, key=lambda index: overlap_scores.get(index, 0.0), reverse=True)
        return ranked[: self.lexical_candidate_limit]

    def semantic_scores(self, text: str, candidate_indices: list[int]) -> list[float | None]:
        if self.embedder is None or self.entry_embeddings is None or not candidate_indices:
            return [None] * len(candidate_indices)
        query_embedding = self.embedder.encode([text])[0]
        candidate_embeddings = self.entry_embeddings[candidate_indices]
        scores = torch.matmul(candidate_embeddings, query_embedding)
        return [float(score) for score in scores]

    def lexical_score(self, text: str, entry: OnetEntry) -> float:
        normalized = normalize_text(text)
        candidate_texts = (entry.normalized_text, *entry.alias_normalized)
        best_score = 0.0
        query_tokens = set(tokenize(normalized))

        for candidate in candidate_texts:
            if not candidate:
                continue
            exact = 1.0 if normalized == candidate else 0.0
            containment = 1.0 if normalized in candidate or candidate in normalized else 0.0
            sequence = SequenceMatcher(None, normalized, candidate).ratio()
            candidate_tokens = set(tokenize(candidate))
            jaccard = len(query_tokens & candidate_tokens) / max(len(query_tokens | candidate_tokens), 1)
            score = max(exact, 0.45 * sequence + 0.35 * jaccard + 0.20 * containment)
            best_score = max(best_score, score)
        return best_score

    def map_entity(
        self,
        *,
        label: str,
        text: str,
        contextual_onetsoc_codes: set[str] | None = None,
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> list[OnetMatch]:
        target_types = set(self.supported_targets(label))
        if not target_types:
            return []

        candidate_indices = self.candidate_indices(text, target_types)
        semantic_scores = self.semantic_scores(text, candidate_indices)
        matches: list[OnetMatch] = []
        for candidate_index, semantic_score in zip(candidate_indices, semantic_scores):
            entry = self.entries[candidate_index]
            lexical = self.lexical_score(text, entry)
            context_boost = 0.0
            if contextual_onetsoc_codes and entry.onetsoc_code in contextual_onetsoc_codes:
                context_boost += 0.08
            if entry.entry_type == "technology_skill" and entry.hot_technology:
                context_boost += 0.03
            if entry.entry_type == "technology_skill" and entry.in_demand:
                context_boost += 0.03

            semantic_component = semantic_score if semantic_score is not None else lexical
            score = 0.55 * lexical + 0.45 * semantic_component + context_boost
            if score < min_score:
                continue
            matches.append(
                OnetMatch(
                    entry=entry,
                    score=score,
                    lexical_score=lexical,
                    semantic_score=semantic_score,
                    context_boost=context_boost,
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:top_k]


def map_record_entities(
    row: dict[str, Any],
    mapper: OnetMapper,
    *,
    top_k: int = 5,
    min_score: float = 0.35,
) -> list[dict[str, Any]]:
    entities = [entity for entity in (row.get("entities") or []) if isinstance(entity, dict)]
    role_context_codes: set[str] = set()
    role_mappings: list[dict[str, Any]] = []

    for entity_index, entity in enumerate(entities):
        label = stringify(entity.get("label")).upper()
        if label != "JOB_ROLE":
            continue
        matches = mapper.map_entity(
            label=label,
            text=stringify(entity.get("text")),
            contextual_onetsoc_codes=None,
            top_k=top_k,
            min_score=min_score,
        )
        if matches:
            role_context_codes.add(matches[0].entry.onetsoc_code)
        role_mappings.append(
            {
                "entity_index": entity_index,
                "entity_label": label,
                "entity_text": stringify(entity.get("text")),
                "start": entity.get("start"),
                "end": entity.get("end"),
                "normalized": stringify(entity.get("normalized")),
                "supported": bool(mapper.supported_targets(label)),
                "target_entry_types": list(mapper.supported_targets(label)),
                "candidates": [match.to_dict() for match in matches],
            }
        )

    mappings_by_index = {payload["entity_index"]: payload for payload in role_mappings}

    for entity_index, entity in enumerate(entities):
        if entity_index in mappings_by_index:
            continue
        label = stringify(entity.get("label")).upper()
        matches = mapper.map_entity(
            label=label,
            text=stringify(entity.get("text")),
            contextual_onetsoc_codes=role_context_codes,
            top_k=top_k,
            min_score=min_score,
        )
        mappings_by_index[entity_index] = {
            "entity_index": entity_index,
            "entity_label": label,
            "entity_text": stringify(entity.get("text")),
            "start": entity.get("start"),
            "end": entity.get("end"),
            "normalized": stringify(entity.get("normalized")),
            "supported": bool(mapper.supported_targets(label)),
            "target_entry_types": list(mapper.supported_targets(label)),
            "candidates": [match.to_dict() for match in matches],
        }

    return [mappings_by_index[index] for index in sorted(mappings_by_index)]


def summarize_mappings(mapped_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total_entities = 0
    supported_entities = 0
    mapped_entities = 0
    by_label: dict[str, dict[str, int]] = {}

    for row in mapped_rows:
        for mapping in row.get("onet_mappings") or []:
            total_entities += 1
            label = stringify(mapping.get("entity_label")).upper()
            stats = by_label.setdefault(label, {"entities": 0, "supported": 0, "mapped": 0})
            stats["entities"] += 1
            if mapping.get("supported"):
                supported_entities += 1
                stats["supported"] += 1
            if mapping.get("candidates"):
                mapped_entities += 1
                stats["mapped"] += 1

    return {
        "total_entities": total_entities,
        "supported_entities": supported_entities,
        "mapped_entities": mapped_entities,
        "mapped_rate": mapped_entities / max(supported_entities, 1),
        "by_label": by_label,
    }


def summarize_record_mappings(mappings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    supported_entities = 0
    mapped_entities = 0
    best_score_total = 0.0
    best_score_count = 0
    role_codes: list[str] = []
    code_counter: Counter[str] = Counter()
    by_label: dict[str, dict[str, int]] = {}

    for mapping in mappings:
        label = stringify(mapping.get("entity_label")).upper()
        stats = by_label.setdefault(label, {"entities": 0, "supported": 0, "mapped": 0})
        stats["entities"] += 1

        if mapping.get("supported"):
            supported_entities += 1
            stats["supported"] += 1

        candidates = [candidate for candidate in (mapping.get("candidates") or []) if isinstance(candidate, dict)]
        if not candidates:
            continue

        mapped_entities += 1
        stats["mapped"] += 1

        best = candidates[0]
        score = best.get("score")
        if isinstance(score, (int, float)):
            best_score_total += float(score)
            best_score_count += 1

        onetsoc_code = stringify(best.get("onetsoc_code"))
        if onetsoc_code:
            code_counter[onetsoc_code] += 1
            if label == "JOB_ROLE" and onetsoc_code not in role_codes:
                role_codes.append(onetsoc_code)

    top_codes = [
        {"onetsoc_code": code, "count": count}
        for code, count in code_counter.most_common(5)
    ]
    return {
        "supported_entities": supported_entities,
        "mapped_entities": mapped_entities,
        "mapped_rate": mapped_entities / max(supported_entities, 1),
        "average_best_score": best_score_total / max(best_score_count, 1),
        "top_onetsoc_codes": top_codes,
        "role_onetsoc_codes": role_codes,
        "by_label": by_label,
    }
