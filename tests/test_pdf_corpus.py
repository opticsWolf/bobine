"""Integration tests over the real test-PDF corpus (bobine[pdf-ingest]).

Fixtures: ``tests/fixtures/pdf/*.pdf`` — trimmed CC BY 4.0 arXiv papers
(see ``SOURCES.md``) plus a generated scanned page. These tests prove real
recognition, not just that the models load: born-digital text extraction,
SURGICAL formula recognition, layout→OCR/table on real pages, and the
scanned-document path.

All gated behind ``integration`` (and ``slow`` where ONNX runs over many
pages). Requires ``pip install -e ".[pdf-ingest,formula]"``.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

FIXTURES = Path(__file__).parent / "fixtures" / "pdf"
SOLITONS = FIXTURES / "solitons.pdf"
SPLITTING = FIXTURES / "splitting_methods.pdf"
TRUST_ML = FIXTURES / "trust_ml.pdf"
SCANNED = FIXTURES / "scanned_page.pdf"


def _convert(name: Path, mode, tmp_path, **kw):
    from bobine import ConverterConfig, ingest_document

    cfg = ConverterConfig(routing_mode=mode, use_onnx=True, render_dpi=150, **kw)
    return ingest_document(name, tmp_path / "out", config=cfg, lint=False)


class TestBornDigitalText:
    @pytest.mark.slow
    def test_never_mode_extracts_full_text(self, tmp_path):
        pytest.importorskip("pdf_oxide")
        from bobine import RoutingMode

        r = _convert(SOLITONS, RoutingMode.NEVER, tmp_path)
        md = r.md_path.read_text(encoding="utf-8")
        assert r.page_count == 4
        assert "Vector Edge Solitons" in md  # title survived the text layer
        assert len(md) > 1000

    @pytest.mark.slow
    def test_surgical_recognizes_formulas(self, tmp_path):
        """Real LaTeX pages: math boxes detected AND recognized to LaTeX."""
        pytest.importorskip("pdf_oxide")
        from bobine import RoutingMode

        r = _convert(SPLITTING, RoutingMode.SURGICAL, tmp_path)
        md = r.md_path.read_text(encoding="utf-8")
        assert r.page_count == 5
        assert "$$" in md  # at least one display-formula block spliced in
        assert "\\frac" in md or "\\int" in md or "\\sum" in md

    @pytest.mark.slow
    def test_headings_detected(self, tmp_path):
        pytest.importorskip("pdf_oxide")
        from bobine import RoutingMode

        r = _convert(SPLITTING, RoutingMode.SURGICAL, tmp_path)
        md = r.md_path.read_text(encoding="utf-8")
        assert any(line.startswith("# ") for line in md.splitlines())


class TestFullStructureOnnx:
    @pytest.mark.slow
    def test_scanned_page_routes_to_layout_ocr(self, tmp_path):
        """Raster page (no text layer) → ONNX layout+OCR produces text."""
        pytest.importorskip("pdf_oxide")
        from bobine import RoutingMode

        r = _convert(SCANNED, RoutingMode.SURGICAL, tmp_path)
        md = r.md_path.read_text(encoding="utf-8")
        assert r.page_count == 1
        # OCR should read our synthetic words back (garbled order acceptable)
        assert "recognition" in md or "pipeline" in md or "document" in md
        assert len(md) > 200

    @pytest.mark.slow
    def test_table_and_figures(self, tmp_path):
        """ML paper pages: table → HTML, figures → staged assets."""
        pytest.importorskip("pdf_oxide")
        from bobine import RoutingMode

        r = _convert(TRUST_ML, RoutingMode.ALWAYS, tmp_path)
        md = r.md_path.read_text(encoding="utf-8")
        assert "<table" in md  # rapid_table produced HTML
        assert "okf-asset://" in md  # figures staged as assets
        assert len(md) > 5000
