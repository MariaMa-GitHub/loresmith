"""Render an HTML table node as pipe-Markdown."""

from __future__ import annotations


def render_table(node) -> str:
    """Convert a selectolax table Node to a pipe-Markdown string.

    Each row becomes ``| cell | cell |``. A separator row is inserted after
    the first row. Rows with no cells are skipped.
    """
    rows = node.css("tr")
    lines: list[str] = []
    for i, row in enumerate(rows):
        cells = row.css("th, td")
        if not cells:
            continue
        cell_texts = [c.text(strip=True) for c in cells]
        pipe_row = "| " + " | ".join(cell_texts) + " |"
        lines.append(pipe_row)
        if len(lines) == 1:
            sep = "| " + " | ".join("---" for _ in cell_texts) + " |"
            lines.append(sep)
    return "\n".join(lines)
