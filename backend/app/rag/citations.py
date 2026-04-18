from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

_INLINE_CITATION_GROUP_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_PARAGRAPH_SPLIT_RE = re.compile(r"(\n\s*\n+)")


@dataclass(frozen=True)
class NormalizedCitationBundle:
    answer: str
    citations: list[dict[str, Any]]


def _parse_citation_group(raw_group: str) -> list[int]:
    return [int(raw_index.strip()) for raw_index in raw_group.split(",")]


def _sanitize_citation_payload(
    original_index: int,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        source_url = str(payload["source_url"])
    except (KeyError, TypeError, ValueError):
        return None
    if not source_url:
        return None

    normalized: dict[str, Any] = {
        "index": original_index,
        "source_url": source_url,
        "title": infer_source_title(payload),
    }
    passage_id = payload.get("passage_id")
    if passage_id is not None:
        normalized["passage_id"] = passage_id
    return normalized


def _derive_title_from_content(content: str) -> str | None:
    stripped = content.lstrip()
    if not stripped:
        return None

    if "\n\n" in stripped:
        candidate = stripped.split("\n\n", maxsplit=1)[0].strip()
        if candidate and len(candidate) <= 120:
            return candidate
    return None


def _humanize_url_segment(raw: str) -> str:
    decoded = unquote(raw).strip()
    decoded = decoded.replace("_", " ").replace("-", " ")
    return " ".join(decoded.split())


def infer_source_title(payload: Mapping[str, Any]) -> str:
    raw_title = payload.get("title")
    if isinstance(raw_title, str) and raw_title.strip():
        return " ".join(raw_title.split())

    raw_content = payload.get("content")
    if isinstance(raw_content, str):
        derived = _derive_title_from_content(raw_content)
        if derived:
            return derived

    raw_source_url = payload.get("source_url")
    if isinstance(raw_source_url, str) and raw_source_url.strip():
        parsed = urlparse(raw_source_url)
        path = parsed.path.rstrip("/")
        if "/wiki/" in path:
            candidate = path.rsplit("/wiki/", maxsplit=1)[-1]
        else:
            candidate = path.rsplit("/", maxsplit=1)[-1]
        humanized = _humanize_url_segment(candidate)
        if humanized:
            return humanized

    return "Source"


def strip_inline_citations(text: str) -> str:
    """Remove inline citation groups such as ``[1]`` or ``[2, 4]``."""
    return _INLINE_CITATION_GROUP_RE.sub(" ", text)


def parse_inline_citation_indices(text: str) -> list[int]:
    """Return unique citation indices in the order they first appear."""
    seen: set[int] = set()
    indices: list[int] = []

    for match in _INLINE_CITATION_GROUP_RE.finditer(text):
        for idx in _parse_citation_group(match.group(1)):
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)

    return indices


def normalize_answer_citations(
    answer: str,
    *,
    passages: Sequence[Mapping[str, Any]] | None = None,
    citations: Sequence[Mapping[str, Any]] | None = None,
) -> NormalizedCitationBundle:
    """Apply the Loresmith citation policy to a generated answer.

    Policy:
    - Citation identity is the source document (``source_url``), not a chunk.
    - Citation numbers are dense ``1..N`` in first-reference order.
    - Each factual paragraph ends with at most one citation cluster.
    - Repeated same-source citations within a paragraph collapse to one index.
    """
    citation_lookup: dict[int, dict[str, Any]] = {}

    for citation in citations or []:
        try:
            original_index = int(citation["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if original_index in citation_lookup:
            continue
        normalized = _sanitize_citation_payload(original_index, citation)
        if normalized is not None:
            citation_lookup[original_index] = normalized

    for original_index, passage in enumerate(passages or [], start=1):
        if original_index in citation_lookup:
            continue
        normalized = _sanitize_citation_payload(original_index, passage)
        if normalized is not None:
            citation_lookup[original_index] = normalized

    if not citation_lookup:
        return NormalizedCitationBundle(answer=answer, citations=[])

    dense_index_map: dict[int, int] = {}
    dense_index_by_source_url: dict[str, int] = {}
    normalized_citations: list[dict[str, Any]] = []

    def _dense_index_for(original_index: int) -> int | None:
        if original_index in dense_index_map:
            return dense_index_map[original_index]
        citation = citation_lookup.get(original_index)
        if citation is None:
            return None
        source_url = citation["source_url"]
        dense_index = dense_index_by_source_url.get(source_url)
        if dense_index is not None:
            dense_index_map[original_index] = dense_index
            return dense_index

        dense_index = len(normalized_citations) + 1
        dense_index_map[original_index] = dense_index
        dense_index_by_source_url[source_url] = dense_index
        normalized_citation = dict(citation)
        normalized_citation["index"] = dense_index
        normalized_citations.append(normalized_citation)
        return dense_index

    def _normalize_paragraph(paragraph: str) -> str:
        dense_indices: set[int] = set()

        def _strip_and_collect(match: re.Match[str]) -> str:
            for original_index in _parse_citation_group(match.group(1)):
                dense_index = _dense_index_for(original_index)
                if dense_index is not None:
                    dense_indices.add(dense_index)
            return ""

        stripped = _INLINE_CITATION_GROUP_RE.sub(_strip_and_collect, paragraph)
        if not dense_indices:
            return paragraph

        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" *\n *", "\n", cleaned)
        cleaned = cleaned.strip()

        citation_cluster = "".join(f"[{index}]" for index in sorted(dense_indices))
        if not cleaned:
            return citation_cluster
        return f"{cleaned} {citation_cluster}"

    parts = _PARAGRAPH_SPLIT_RE.split(answer)
    normalized_answer = "".join(
        part if _PARAGRAPH_SPLIT_RE.fullmatch(part) else _normalize_paragraph(part)
        for part in parts
    )
    return NormalizedCitationBundle(
        answer=normalized_answer,
        citations=normalized_citations,
    )
