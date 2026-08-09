"""Tests for the text-type document model (bobine.documents)."""

from bobine.documents import Document, load_markdown_document, wrap_thoughts


class TestLoadMarkdownDocument:
    def test_loads_plain_markdown(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("# Hello\n\nBody text.\n", encoding="utf-8")
        doc = load_markdown_document(p)
        assert doc.id == "note"
        assert doc.title == "note"
        assert doc.description == ""
        assert "Body text." in doc.body
        assert doc.type == "note"
        assert doc.source_path == p

    def test_missing_file_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_markdown_document(tmp_path / "nope.md")

    def test_concept_id_slugified(self, tmp_path):
        p = tmp_path / "My Note.md"
        p.write_text("content", encoding="utf-8")
        doc = load_markdown_document(p)
        assert doc.id == "my_note"

    def test_explicit_overrides(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("content", encoding="utf-8")
        doc = load_markdown_document(p, concept_id="cid", title="T", description="D", tags=["a"])
        assert doc.id == "cid"
        assert doc.title == "T"
        assert doc.description == "D"
        assert doc.tags == ["a"]

    def test_metadata_precedence(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text(
            "---\ntitle: FM Title\ntype: paper\ntags: [x, y]\ndescription: FM desc\n---\n\nbody",
            encoding="utf-8",
        )
        doc = load_markdown_document(p, title="Kw Title")
        assert doc.title == "Kw Title"  # explicit wins
        assert doc.type == "paper"  # frontmatter fills the rest
        assert doc.tags == ["x", "y"]
        assert doc.description == "FM desc"


class TestWrapThoughts:
    def test_builds_thought_document(self):
        doc = wrap_thoughts("some reasoning", topic="graphs")
        assert doc.type == "thought"
        assert doc.id.startswith("thought_graphs_")
        assert doc.title == "Thought: graphs"
        assert "thought_type: reasoning" in doc.body
        assert "some reasoning" in doc.body
        assert doc.tags == ["thought", "reasoning", "graphs"]

    def test_explicit_concept_id(self):
        doc = wrap_thoughts("t", topic="g", concept_id="custom")
        assert doc.id == "custom"

    def test_extra_tags_deduplicated(self):
        doc = wrap_thoughts("t", topic="g", tags=["thought", "extra"])
        assert doc.tags == ["thought", "reasoning", "g", "extra"]


class TestDocumentToOKF:
    def test_round_trip_preserves_body(self):
        doc = Document(id="x", title="X", description="", body="# B\n\nbody", type="note")
        md = doc.to_okf_markdown()
        assert "body" in md
