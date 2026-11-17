import pathlib

HTML = (pathlib.Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")


def test_citation_linker_is_wired_in():
    assert "function linkCitations" in HTML
    assert "linkCitations();" in HTML
    assert 'item.id = "trail-"' in HTML


def test_highlighter_skips_citation_links():
    assert 'closest(".cite")' in HTML