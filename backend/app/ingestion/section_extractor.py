"""Extract structured sections from MediaWiki HTML pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from selectolax.parser import HTMLParser

from app.ingestion.table_renderer import render_table

_NOISE_SELECTORS = [
    ".toc",
    ".navbox",
    ".references",
    ".reflist",
    ".noprint",
    "script",
    "style",
    ".mw-editsection",
]

_BLOCK_TAGS = {"p", "ul", "ol", "dl", "pre"}


@dataclass(frozen=True)
class SectionRecord:
    section_path: tuple[str, ...]
    text: str
    kind: Literal["prose", "table"]


def extract_sections(html: str) -> list[SectionRecord]:
    """Parse HTML and return a flat list of SectionRecords.

    Headings (h2/h3/h4) define the section path. Tables are rendered to
    pipe-Markdown and emitted as ``kind="table"`` records. Prose text from
    block elements is accumulated and flushed on heading transitions, table
    encounters, and end-of-document.
    """
    parser = HTMLParser(html)

    content_node = (
        parser.css_first(".mw-parser-output")
        or parser.css_first("#mw-content-text")
        or parser.css_first("body")
    )
    if content_node is None:
        return []

    for selector in _NOISE_SELECTORS:
        for node in content_node.css(selector):
            node.decompose()

    records: list[SectionRecord] = []
    current_path: list[str] = []
    current_text: list[str] = []

    def flush_prose() -> None:
        text = "\n".join(current_text).strip()
        if text:
            records.append(
                SectionRecord(
                    section_path=tuple(current_path),
                    text=text,
                    kind="prose",
                )
            )
        current_text.clear()

    # Iterate only direct children (not descendants) so table cell text never
    # leaks into prose records. The child/next linked-list traversal is O(N)
    # over direct children without recursing into subtrees.
    node = content_node.child
    while node is not None:
        tag = getattr(node, "tag", None)
        next_node = node.next  # capture before any decompose
        if tag in {"h2", "h3", "h4"}:
            flush_prose()
            heading_text = node.text(strip=True)
            if tag == "h2":
                current_path = [heading_text]
            elif tag == "h3":
                current_path = current_path[:1] + [heading_text]
            elif tag == "h4":
                current_path = current_path[:2] + [heading_text]
        elif tag == "table":
            flush_prose()
            md = render_table(node)
            if md.strip():
                records.append(
                    SectionRecord(
                        section_path=tuple(current_path),
                        text=md,
                        kind="table",
                    )
                )
        elif tag in _BLOCK_TAGS:
            text = node.text(strip=True)
            if text:
                current_text.append(text)
        node = next_node

    flush_prose()
    return records
