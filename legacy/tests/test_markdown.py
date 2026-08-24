"""Tests for markdown linting (bobine.markdown).

mordant may not be installed in this environment; the no-op passthrough path
must be correct either way.
"""

from bobine.markdown import lint_markdown, lint_markdown_file


class TestLintMarkdown:
    def test_clean_markdown_passthrough(self):
        result = lint_markdown("# Title\n\nBody.\n")
        assert result["content"] == "# Title\n\nBody.\n"
        assert result["fixed"] is False

    def test_returns_expected_shape(self):
        result = lint_markdown("hello")
        for key in ("content", "fixed", "fixed_count", "unfixable", "errors"):
            assert key in result

    def test_missing_mordant_degrades_gracefully(self, monkeypatch):
        import bobine.markdown as md

        monkeypatch.setattr(md, "mordant", None)
        result = lint_markdown("# ok")
        assert result["content"] == "# ok"
        assert result["fixed"] is False
        # a diagnostic notes the linter is unavailable
        assert len(result["unfixable"]) == 1
        assert "mordant" in str(result["unfixable"][0])

    def test_lint_file_writes_back_when_fixed(self, tmp_path, monkeypatch):
        import bobine.markdown as md

        if md.mordant is None:
            import pytest

            pytest.skip("mordant not installed")

        p = tmp_path / "a.md"
        p.write_text("line1   \n\n\nline2", encoding="utf-8")  # MD009/MD012
        result = lint_markdown_file(p)
        assert result["fixed_count"] >= 1
        rewritten = p.read_text(encoding="utf-8")
        assert rewritten == result["content"]

    def test_lint_file_untouched_when_clean(self, tmp_path):
        p = tmp_path / "b.md"
        p.write_text("# Title\n\nbody.\n", encoding="utf-8")
        result = lint_markdown_file(p)
        assert result["fixed"] is False
        assert p.read_text(encoding="utf-8") == "# Title\n\nbody.\n"
