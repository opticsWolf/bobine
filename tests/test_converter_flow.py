"""Fake-backend tests for HybridConverter's core logic.

Drives the real converter code (routing, math signal, scanned detection,
formula boxes, splicing, code blocks, ONNX page assembly, image extraction)
using fake pdf_oxide objects from ``conftest`` — no native dependencies needed.
"""

from __future__ import annotations

from pathlib import Path

from conftest import PNG, FakeChar, FakePage, FakePdfDocument
from PIL import Image

from bobine.config import ConverterConfig, RoutingMode
from bobine.converter import HybridConverter


def _conv(config: ConverterConfig | None = None) -> HybridConverter:
    return HybridConverter(config or ConverterConfig(), log=lambda *a, **k: None)


# ── fast path ──────────────────────────────────────────────────────────────


class TestFastPath:
    def test_markdown_used_first(self):
        page = FakePage(text="plain", markdown_text="# Markdown!")
        conv = _conv(ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False))
        assert conv._fast_page_markdown(page) == "# Markdown!"

    def test_falls_back_to_plain_text_and_attr(self):
        class _NoMarkdown(FakePage):
            def markdown(self, detect_headings=True):
                raise AttributeError("no markdown api")

        p1 = _NoMarkdown(text="plain fallback")
        assert _conv()._fast_page_markdown(p1) == "plain fallback"

        p2 = FakePage(text="attr text")
        p2.markdown = None  # type: ignore[assignment]  # attribute, not method
        p3 = FakePage(text="")
        p3.text = ""  # empty → "" via getattr path
        assert _conv()._fast_page_markdown(p2) in ("attr text", "")
        assert _conv()._fast_page_markdown(p3) == ""

    def test_convert_pdf_never_mode(self, tmp_path, monkeypatch):
        doc = FakePdfDocument(
            pages=[
                FakePage.from_text("Page one.", index=0),
                FakePage.from_text("Page two.", index=1),
            ]
        )
        cfg = ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False)
        conv = _conv(cfg)

        # Route through convert_pdf by writing a fake path + patching PdfDocument
        import bobine.converter as conv_mod

        monkeypatch.setattr(conv_mod, "PdfDocument", lambda _path: doc)  # context manager
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        md = conv.convert_pdf(
            pdf,
            tmp_path / "work",
            should_continue=lambda: True,
            on_page=lambda i, n: None,
        )

        assert "Page one." in md
        assert "Page two." in md
        assert "---" in md  # page separator

    def test_should_continue_stops_early(self, tmp_path, monkeypatch):
        doc = FakePdfDocument(
            pages=[
                FakePage.from_text("A", index=0),
                FakePage.from_text("B", index=1),
                FakePage.from_text("C", index=2),
            ]
        )
        cfg = ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False)
        conv = _conv(cfg)
        import bobine.converter as conv_mod

        monkeypatch.setattr(conv_mod, "PdfDocument", lambda _path: doc)
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        seen = []
        md = conv.convert_pdf(
            pdf,
            tmp_path / "work",
            should_continue=lambda: len(seen) < 2,
            on_page=lambda i, n: seen.append((i, n)),
        )

        assert "A" in md and "B" in md and "C" not in md
        assert seen == [(0, 3), (1, 3)]

    def test_on_page_receives_totals(self, tmp_path, monkeypatch):
        doc = FakePdfDocument(pages=[FakePage(index=0), FakePage(index=1), FakePage(index=2)])
        conv = _conv(ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False))
        import bobine.converter as conv_mod

        monkeypatch.setattr(conv_mod, "PdfDocument", lambda _path: doc)
        pdf = tmp_path / "d.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        seen = []
        conv.convert_pdf(pdf, tmp_path / "w", lambda: True, lambda i, n: seen.append((i, n)))
        assert seen == [(0, 3), (1, 3), (2, 3)]


# ── routing signals ────────────────────────────────────────────────────────


class TestRoutingSignals:
    def test_math_signal_from_fonts(self):
        chars = [
            FakeChar("x", font_name="cmmi10", bbox=(0, 0, 5, 10)),
            FakeChar("+", font_name="Helvetica", bbox=(5, 0, 10, 10)),
        ]
        page = FakePage(text="x+", chars=chars)
        math, total = _conv()._page_math_signal(page)
        assert math == 1
        assert total == 2

    def test_math_signal_from_unicode(self):
        page = FakePage(text="α + β", chars=[])
        math, total = _conv()._page_math_signal(page)
        assert math == 2  # α and β are Greek
        assert total == 5

    def test_is_scanned_image_only_page(self):
        page = FakePage.with_images(n=1)  # no chars
        assert _conv()._is_scanned(page) is True

    def test_is_scanned_text_page(self):
        page = FakePage.from_text("A fully readable sentence of text.")
        assert _conv()._is_scanned(page) is False

    def test_needs_paddle_auto_math(self):
        cfg = ConverterConfig(routing_mode=RoutingMode.AUTO, math_char_threshold=30)
        conv = _conv(cfg)
        math_chars = [FakeChar("α", font_name="cmmi10") for _ in range(40)]
        page = FakePage(text="α" * 40, chars=math_chars)
        assert conv._needs_paddle(page) is True

    def test_needs_paddle_auto_plain(self):
        cfg = ConverterConfig(routing_mode=RoutingMode.AUTO, math_char_threshold=30)
        conv = _conv(cfg)
        page = FakePage.from_text("This is a completely ordinary page of prose text.")
        assert conv._needs_paddle(page) is False

    def test_needs_paddle_never_false(self):
        conv = _conv(ConverterConfig(routing_mode=RoutingMode.NEVER))
        page = FakePage.with_images(n=1)
        assert conv._needs_paddle(page) is False

    def test_needs_paddle_always_true(self):
        conv = _conv(ConverterConfig(routing_mode=RoutingMode.ALWAYS))
        assert conv._needs_paddle(FakePage()) is True


# ── formula boxes ──────────────────────────────────────────────────────────


class TestFormulaBoxes:
    def test_math_boxes_from_chars_merges(self):
        # Two math glyphs close together → one merged box
        chars = [
            FakeChar("α", font_name="cmmi10", bbox=(100, 700, 108, 712)),
            FakeChar("β", font_name="cmmi10", bbox=(109, 700, 117, 712)),
        ]
        page = FakePage(text="αβ", chars=chars)
        boxes = _conv(ConverterConfig(min_formula_math_chars=2))._math_boxes_from_chars(page)
        assert len(boxes) == 1
        x0, _y0, x1, _y1 = boxes[0]
        assert x0 == 100 and x1 == 117

    def test_math_boxes_split_when_far_apart(self):
        chars = [
            FakeChar("α", font_name="cmmi10", bbox=(100, 700, 108, 712)),
            FakeChar("β", font_name="cmmi10", bbox=(500, 700, 508, 712)),
        ]
        page = FakePage(text="αβ", chars=chars)
        boxes = _conv(ConverterConfig(min_formula_math_chars=1))._math_boxes_from_chars(page)
        assert len(boxes) == 2

    def test_no_math_chars_no_boxes(self):
        page = FakePage.from_text("No math here.")
        assert _conv()._math_boxes_from_chars(page) == []


# ── code blocks ────────────────────────────────────────────────────────────


class TestCodeBlocks:
    def test_wraps_mono_runs(self):
        chars = [
            FakeChar("d", font_name="Courier New", bbox=(0, 700, 8, 712)),
            FakeChar("e", font_name="Courier New", bbox=(8, 700, 16, 712)),
            FakeChar("f", font_name="Courier New", bbox=(16, 700, 24, 712)),
            FakeChar("g", font_name="Courier New", bbox=(24, 700, 32, 712)),
            FakeChar("h", font_name="Courier New", bbox=(32, 700, 40, 712)),
            FakeChar("i", font_name="Arial", bbox=(50, 700, 58, 712)),
        ]
        page = FakePage(text="defghi", chars=chars)
        blocks = _conv()._wrap_code_blocks(page)
        assert len(blocks) == 1
        assert blocks[0] == "```\ndefgh\n```"

    def test_disabled_when_flag_off(self):
        page = FakePage.from_text("x", font="Courier New")
        cfg = ConverterConfig(detect_code_blocks=False)
        assert _conv(cfg)._wrap_code_blocks(page) == []


# ── splicing ───────────────────────────────────────────────────────────────


class TestSplice:
    def test_direct_replace(self):
        conv = _conv()
        md = "Text with a formula α here."
        assert conv._splice(md, [("α", r"$\alpha$")]) == r"Text with a formula $\alpha$ here."

    def test_whitespace_tolerant_replace(self):
        conv = _conv()
        md = "Text with\na   formula here."
        out = conv._splice(md, [("with a formula", "**block**")])
        assert "**block**" in out

    def test_leftovers_appended(self):
        conv = _conv()
        md = "Nothing here."
        out = conv._splice(md, [("missing needle", "$$\n\\int x dx\n$$")])
        assert "\\int x dx" in out  # appended at the end


# ── image extraction ───────────────────────────────────────────────────────


class TestImageExtraction:
    def test_extract_page_images(self, tmp_path):
        doc = FakePdfDocument()
        doc.extracted[0] = [{"data": PNG, "format": "png"}]
        page = FakePage(index=0)
        paths = _conv()._extract_page_images(doc, page, 0, tmp_path)
        assert len(paths) == 1
        assert paths[0].name == "p0_img0.png"
        assert paths[0].read_bytes() == PNG

    def test_missing_api_returns_empty(self, tmp_path):
        class DocNoExtract:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        paths = _conv()._extract_page_images(DocNoExtract(), FakePage(), 0, tmp_path)
        assert paths == []

    def test_gallery_appended_for_unreferenced(self, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        doc = FakePdfDocument(pages=[FakePage.from_text("no images referenced", index=0)])
        doc.extracted[0] = [{"data": PNG, "format": "png"}]
        cfg = ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False, extract_images=True)
        conv = _conv(cfg)
        import bobine.converter as conv_mod

        monkeypatch.setattr(conv_mod, "PdfDocument", lambda _p: doc)
        pdf = tmp_path / "g.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        md = conv.convert_pdf(pdf, work, lambda: True, lambda i, n: None)
        assert "![](p0_img0.png)" in md


# ── ONNX page assembly with injected fake engines ─────────────────────────


class TestFullStructurePage:
    @staticmethod
    def _page_and_conv(tmp_path, cfg=None):
        img = Image.new("RGB", (300, 400), "white")
        page = FakePage(text="", chars=[], width=612.0, height=792.0, render_result=img)
        conv = _conv(cfg or ConverterConfig(routing_mode=RoutingMode.ALWAYS))
        return page, conv

    def test_text_region_ocr(self, tmp_path):
        page, conv = self._page_and_conv(tmp_path)
        conv.rapid.layout_regions = lambda img: [([50, 600, 300, 700], "text", 0.9)]
        conv.rapid.ocr_lines = lambda img: [((0, 0, 1, 1), "Hello from OCR", 0.95)]
        md = conv._full_structure_page_markdown(FakePdfDocument(), page, 0, tmp_path)
        assert md is not None and "Hello from OCR" in md

    def test_title_region_becomes_heading(self, tmp_path):
        page, conv = self._page_and_conv(tmp_path)
        conv.rapid.layout_regions = lambda img: [([50, 600, 300, 700], "title", 0.9)]
        conv.rapid.ocr_lines = lambda img: [((0, 0, 1, 1), "Big Title", 0.95)]
        md = conv._full_structure_page_markdown(FakePdfDocument(), page, 0, tmp_path)
        assert md is not None and "## Big Title" in md

    def test_table_region_converted_to_gfm(self, tmp_path):
        page, conv = self._page_and_conv(tmp_path)
        conv.rapid.layout_regions = lambda img: [([50, 600, 300, 700], "table", 0.9)]
        conv.rapid.table_html = lambda img: (
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        )
        conv.rapid.ocr_lines = lambda img: []
        md = conv._full_structure_page_markdown(FakePdfDocument(), page, 0, tmp_path)
        assert md is not None and "| A | B |" in md

    def test_formula_region_wrapped_in_display(self, tmp_path):
        page, conv = self._page_and_conv(tmp_path)
        conv.rapid.layout_regions = lambda img: [([50, 600, 300, 700], "formula", 0.9)]
        conv.rapid.recognize_formula = lambda p: r"\sum_{i=1}^{n} x_i"
        md = conv._full_structure_page_markdown(FakePdfDocument(), page, 0, tmp_path)
        assert md is not None
        assert md.strip().startswith("$$\n") and md.strip().endswith("\n$$")  # display LaTeX
        assert r"\sum_{i=1}^{n} x_i" in md

    def test_figure_region_emits_link(self, tmp_path):
        page, conv = self._page_and_conv(tmp_path)
        conv.rapid.layout_regions = lambda img: [([50, 600, 300, 700], "figure", 0.9)]
        md = conv._full_structure_page_markdown(FakePdfDocument(), page, 0, tmp_path)
        assert md is not None and "![](" in md

    def test_reading_order_y_then_x(self, tmp_path):
        page, conv = self._page_and_conv(tmp_path)
        # lower y (higher on page, since y-up) comes first
        conv.rapid.layout_regions = lambda img: [
            ([10, 100, 100, 200], "text", 0.9),  # lower on page → first
            ([10, 500, 100, 600], "text", 0.9),  # higher on page → second
        ]
        calls = []

        def fake_ocr(img):
            calls.append(img)
            return [((0, 0, 1, 1), f"region{len(calls)}", 1.0)]

        conv.rapid.ocr_lines = fake_ocr
        md = conv._full_structure_page_markdown(FakePdfDocument(), page, 0, tmp_path)
        assert md is not None
        assert md.index("region1") < md.index("region2")

    def test_render_unavailable_falls_back_none(self, tmp_path):
        page = FakePage(text="", chars=[], render_result=None)
        conv = _conv(ConverterConfig(routing_mode=RoutingMode.ALWAYS))
        assert conv._full_structure_page_markdown(FakePdfDocument(), page, 0, tmp_path) is None


class TestSurgicalPage:
    def test_surgical_splices_latex(self, tmp_path):
        img = Image.new("RGB", (200, 260), "white")
        math_chars = [
            FakeChar("α", font_name="cmmi10", bbox=(100, 700, 108, 712)),
            FakeChar("β", font_name="cmmi10", bbox=(109, 700, 117, 712)),
        ]
        page = FakePage(
            text="αβ",
            chars=math_chars,
            width=612.0,
            height=792.0,
            render_result=img,
            markdown_text="A formula αβ here.",
        )
        doc = FakePdfDocument(pages=[page])
        doc.region_texts[0] = "αβ"

        conv = _conv(ConverterConfig(routing_mode=RoutingMode.SURGICAL, min_formula_math_chars=2))
        # fake RapidLaTeXOCR: callable returning (latex, elapse)
        conv.rapid.formula = lambda: lambda data: (r"\alpha\beta", 0.01)

        md = conv._surgical_page_markdown(doc, page, 0, tmp_path)
        assert r"$\alpha\beta$" in md or r"$$\alpha\beta$$" in md

    def test_surgical_no_formula_engine_returns_fast(self, tmp_path):
        page = FakePage.from_text("Just text.", index=0)
        conv = _conv(ConverterConfig(routing_mode=RoutingMode.SURGICAL))
        conv.rapid.formula = lambda: None  # rapid_latex_ocr missing
        md = conv._surgical_page_markdown(FakePdfDocument(), page, 0, tmp_path)
        assert md == "Just text."


# ── routing through _route_page ────────────────────────────────────────────


class TestRoutePage:
    def test_never_fast_path(self):
        page = FakePage.from_text("fast")
        conv = _conv(ConverterConfig(routing_mode=RoutingMode.NEVER))
        assert conv._route_page(None, page, 0, Path(".")) == "fast"

    def test_auto_scanned_goes_full_pipeline(self, tmp_path):
        img = Image.new("RGB", (300, 400), "white")
        page = FakePage(text="", chars=[], images=[{"data": PNG}], render_result=img)
        conv = _conv(ConverterConfig(routing_mode=RoutingMode.AUTO))
        conv.rapid.ocr = lambda: object()  # lazy loader must appear available
        conv.rapid.layout_regions = lambda img: []
        conv.rapid.ocr_lines = lambda img: [((0, 0, 1, 1), "scan text", 0.9)]
        md = conv._route_page(FakePdfDocument(), page, 0, tmp_path)
        assert "scan text" in md
