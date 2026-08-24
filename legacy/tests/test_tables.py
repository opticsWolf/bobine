"""Tests for the HTML table → GFM converter (ported from OKFgraph)."""

from bobine.tables import html_tables_to_gfm


class TestHTMLTablesToGFM:
    def test_simple_table(self):
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        result = html_tables_to_gfm(html)
        assert "| A | B |" in result
        assert "| --- | --- |" in result
        assert "| 1 | 2 |" in result

    def test_table_with_data_rows(self):
        html = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>alpha</td><td>1</td></tr>"
            "<tr><td>beta</td><td>2</td></tr>"
            "</table>"
        )
        result = html_tables_to_gfm(html)
        assert "| Name | Value |" in result
        assert "| alpha | 1 |" in result

    def test_complex_table_with_colspan_kept_as_html(self):
        html = '<table><tr><td colspan="2">merged</td></tr></table>'
        result = html_tables_to_gfm(html)
        assert "<table>" in result or "<td" in result

    def test_empty_table_returns_none(self):
        html = "<table></table>"
        result = html_tables_to_gfm(html)
        assert result == html

    def test_table_with_pipes_escaped(self):
        html = "<table><tr><th>A</th></tr><tr><td>a|b</td></tr></table>"
        result = html_tables_to_gfm(html)
        assert r"\|" in result

    def test_multiple_tables(self):
        html = (
            "<table><tr><th>X</th></tr><tr><td>1</td></tr></table>"
            "\n\nSome text\n\n"
            "<table><tr><th>Y</th></tr><tr><td>2</td></tr></table>"
        )
        result = html_tables_to_gfm(html)
        assert "| X |" in result
        assert "| Y |" in result
