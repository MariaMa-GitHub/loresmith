from app.ingestion.section_extractor import SectionRecord, extract_sections

FIXTURE_HTML = """
<div class="mw-parser-output">
<h2>Overview</h2><p>Zagreus is the player character.</p>
<h2>Mirror of Night</h2>
<h3>Talents</h3>
<p>The Mirror grants various boons.</p>
<h3>Privileged Status</h3>
<p>Grants bonus damage when enemies have two or more status effects.</p>
<h2>Statistics</h2>
<table>
<tr><th>Stat</th><th>Value</th></tr>
<tr><td>HP</td><td>50</td></tr>
</table>
</div>
"""


def test_extract_sections_returns_prose_records():
    records = extract_sections(FIXTURE_HTML)
    overview = next(
        (r for r in records if r.section_path == ("Overview",) and r.kind == "prose"),
        None,
    )
    assert overview is not None
    assert "Zagreus is the player character" in overview.text


def test_extract_sections_nested_headings():
    records = extract_sections(FIXTURE_HTML)
    privileged = next(
        (
            r
            for r in records
            if r.section_path == ("Mirror of Night", "Privileged Status") and r.kind == "prose"
        ),
        None,
    )
    assert privileged is not None
    assert "status effects" in privileged.text


def test_extract_sections_table_record():
    records = extract_sections(FIXTURE_HTML)
    table_rec = next((r for r in records if r.kind == "table"), None)
    assert table_rec is not None
    assert "| Stat | Value |" in table_rec.text
    assert "| HP | 50 |" in table_rec.text


def test_extract_sections_empty_html():
    records = extract_sections('<div class="mw-parser-output"></div>')
    assert records == []
