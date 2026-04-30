from selectolax.parser import HTMLParser

from app.ingestion.table_renderer import render_table

TABLE_HTML = """
<table>
<tr><th>Stat</th><th>Value</th></tr>
<tr><td>HP</td><td>50</td></tr>
<tr><td>Damage</td><td>10</td></tr>
</table>
"""


def test_render_table_produces_pipe_markdown():
    node = HTMLParser(TABLE_HTML).css_first("table")
    assert node is not None
    result = render_table(node)

    lines = [line for line in result.splitlines() if line.strip()]
    assert len(lines) >= 3  # header + separator + at least one data row

    # Header row
    assert "| Stat | Value |" in lines[0]

    # Separator row (second line)
    assert "---" in lines[1]

    # Data rows
    assert "| HP | 50 |" in result
    assert "| Damage | 10 |" in result


def test_render_table_skips_empty_rows():
    html = "<table><tr></tr><tr><td>A</td><td>B</td></tr></table>"
    node = HTMLParser(html).css_first("table")
    assert node is not None
    result = render_table(node)
    lines = [line for line in result.splitlines() if line.strip()]
    # The empty row should be skipped; only the data row (and separator) remain
    assert all("A" in line or "---" in line for line in lines)
