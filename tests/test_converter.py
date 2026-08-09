"""Tests for HybridConverter initialization and heuristics (ported from OKFgraph)."""

import pytest
from conftest import FakeChar, FakePage

from bobine.config import ConverterConfig, RoutingMode
from bobine.converter import HybridConverter, _is_math_unicode, _is_mono_font


class TestHybridConverterInit:
    def test_never_mode_no_models(self):
        cfg = ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False)
        conv = HybridConverter(cfg)
        conv.ensure_models()
        assert conv.rapid._formula is None
        assert conv.rapid._ocr is None
        conv.close()

    def test_converter_close(self):
        cfg = ConverterConfig()
        conv = HybridConverter(cfg)
        conv.close()
        assert conv.rapid._formula is None
        assert conv.rapid._ocr is None
        assert conv.rapid._layout is None
        assert conv.rapid._table is None

    def test_math_unicode_detection(self):
        assert _is_math_unicode("α")  # Greek
        assert _is_math_unicode("∑")  # Math operator
        assert not _is_math_unicode("a")  # Regular ASCII

    def test_mono_font_detection(self):
        assert _is_mono_font("Courier New")
        assert _is_mono_font("Consolas")
        assert not _is_mono_font("Arial")


class TestMathBoxDetection:
    """Regression guards for the real-paper formula-box bugs (2026-08-09).

    Two bugs merged every LaTeX paper's page into one giant box:
    1. ``cmr10`` (Computer Modern Roman = body text) was in the math font
       list — 58% of a real paper's chars were "math".
    2. pdf_oxide >=0.3 reports ``bbox=(x, y, w, h)`` but the merge assumed
       ``(x0, y0, x1, y1)`` — width/height got compared against x/y.
    """

    def test_cmr10_body_text_is_not_math(self):
        from bobine.converter import _MATH_FONT_KEYWORDS

        assert "cmr10" not in _MATH_FONT_KEYWORDS  # body font, not math
        assert "cmmi" in _MATH_FONT_KEYWORDS  # math italic stays
        assert "cmsy" in _MATH_FONT_KEYWORDS  # math symbols stay

    def test_body_text_page_yields_no_boxes(self):
        from bobine.converter import ConverterConfig, HybridConverter

        # A real LaTeX body page: cmr10/cmbx12 prose — must NOT be math.
        page = FakePage(
            chars=[FakeChar("A", font_name="SRNFQE+CMR10"), FakeChar("B", font_name="XZQ+CMBX12")]
            * 30
        )
        conv = HybridConverter(ConverterConfig())
        assert conv._math_boxes_from_chars(page) == []

    def test_math_font_page_yields_tight_boxes_wh_bbox(self):
        from bobine.converter import ConverterConfig, HybridConverter

        # Two distant equations, bbox in pdf_oxide>=0.3 (x, y, w, h) form.
        chars = []
        for _i, x in enumerate(range(50, 150, 10)):
            chars.append(FakeChar("x", font_name="AAA+CMMI10", bbox=(x, 700.0, 8.0, 10.0)))
        for _i, x in enumerate(range(50, 100, 10)):
            chars.append(FakeChar("y", font_name="AAA+CMMI10", bbox=(x, 300.0, 8.0, 10.0)))
        conv = HybridConverter(ConverterConfig())
        boxes = conv._math_boxes_from_chars(FakePage(chars=chars))
        assert len(boxes) == 2  # not merged into one page-spanning box
        top, bottom = sorted(boxes, key=lambda b: b[1])
        assert top[3] < 400 and bottom[1] > 600  # vertical separation preserved
        assert top[2] - top[0] < 60 and bottom[2] - bottom[0] < 120

    def test_xyxy_bbox_form_still_works(self):
        from bobine.converter import ConverterConfig, HybridConverter, _bbox_to_xyxy

        assert _bbox_to_xyxy((100.0, 700.0, 110.0, 710.0)) == (100.0, 700.0, 110.0, 710.0)
        assert _bbox_to_xyxy((100.0, 700.0, 8.0, 10.0)) == (100.0, 700.0, 108.0, 710.0)

        # legacy (x0,y0,x1,y1) bboxes still produce tight boxes
        chars = [FakeChar("x", font_name="CMMI10", bbox=(50.0, 700.0, 60.0, 712.0))] * 6
        conv = HybridConverter(ConverterConfig())
        boxes = conv._math_boxes_from_chars(FakePage(chars=chars))
        assert len(boxes) == 1
        assert boxes[0][2] - boxes[0][0] < 15  # ~one char wide, not page-wide

    # -- line-aware merge (P1) ----------------------------------------------

    def _math_line(self, xs, y, w=8.0, h=10.0, char="x", font="AAA+CMMI10"):
        return [FakeChar(char, font_name=font, bbox=(x, y, w, h)) for x in xs]

    def test_multi_line_equation_merges_into_one_box(self):
        """Two lines of a display equation (12pt apart) → ONE box."""
        from bobine.converter import ConverterConfig, HybridConverter

        chars = self._math_line([50, 60, 70, 80, 90], 700.0) + self._math_line(
            [55, 65, 75, 85, 95], 688.0
        )
        conv = HybridConverter(ConverterConfig(min_formula_math_chars=5))
        boxes = conv._math_boxes_from_chars(FakePage(chars=chars, height=792.0))
        assert len(boxes) == 1
        x0, y0, x1, y1 = boxes[0]
        assert y1 - y0 > 20  # spans both lines
        assert x1 - x0 < 60  # but stays narrow (no page-spanning)

    def test_separate_equations_stay_separate_vertically(self):
        """Equations >1.5 line-heights apart → separate boxes."""
        from bobine.converter import ConverterConfig, HybridConverter

        chars = self._math_line([50, 60, 70, 80, 90], 700.0) + self._math_line(
            [50, 60, 70, 80, 90], 660.0
        )
        conv = HybridConverter(ConverterConfig(min_formula_math_chars=5))
        boxes = conv._math_boxes_from_chars(FakePage(chars=chars, height=792.0))
        assert len(boxes) == 2

    def test_two_columns_not_merged(self):
        """Side-by-side equations (no horizontal overlap) → separate boxes."""
        from bobine.converter import ConverterConfig, HybridConverter

        left = self._math_line([50, 60, 70, 80], 700.0) + self._math_line([50, 60, 70, 80], 688.0)
        right = self._math_line([300, 310, 320, 330], 700.0) + self._math_line(
            [300, 310, 320, 330], 688.0
        )
        conv = HybridConverter(ConverterConfig(min_formula_math_chars=5))
        boxes = conv._math_boxes_from_chars(FakePage(chars=left + right, height=792.0))
        assert len(boxes) == 2

    # -- P2 layout fallback ---------------------------------------------------

    def test_layout_equation_boxes_converts_pixels_to_points(self, monkeypatch):
        pytest.importorskip("numpy")
        from PIL import Image

        from bobine.converter import ConverterConfig, HybridConverter

        conv = HybridConverter(ConverterConfig(formula_layout_fallback=True, render_dpi=144))
        img = Image.new("RGB", (288, 288))
        monkeypatch.setattr(conv, "_render_page_to_pil", lambda *a, **k: (img, 144.0, 144.0))

        class FakeLayout:
            def __call__(self, x):
                return type(
                    "O",
                    (),
                    {"boxes": [[144, 144, 288, 288]], "class_names": ["equation"], "scores": [0.9]},
                )()

        conv.rapid._layout = FakeLayout()
        out = conv._layout_equation_boxes(None, None, 0)
        assert out == [(72.0, 72.0, 144.0, 144.0)]  # pixels / (144/72) → points

    def test_layout_equation_boxes_filters_non_math_labels(self, monkeypatch):
        pytest.importorskip("numpy")
        from PIL import Image

        from bobine.converter import ConverterConfig, HybridConverter

        conv = HybridConverter(ConverterConfig(formula_layout_fallback=True, render_dpi=144))
        img = Image.new("RGB", (288, 288))
        monkeypatch.setattr(conv, "_render_page_to_pil", lambda *a, **k: (img, 144.0, 144.0))

        class FakeLayout:
            def __call__(self, x):
                return type(
                    "O",
                    (),
                    {
                        "boxes": [[10, 10, 20, 20], [30, 30, 40, 40]],
                        "class_names": ["equation", "text"],
                        "scores": [0.9, 0.9],
                    },
                )()

        conv.rapid._layout = FakeLayout()
        out = conv._layout_equation_boxes(None, None, 0)
        assert len(out) == 1  # only the equation label survives

    def test_surgical_fallback_recognizes_layout_boxes(self, monkeypatch, tmp_path):
        """SURGICAL + formula_layout_fallback=True routes layout boxes to the
        formula recognizer when the text layer has no math fonts."""
        from PIL import Image

        from bobine.converter import ConverterConfig, HybridConverter

        page = FakePage(text="some prose", chars=[])  # no math chars
        conv = HybridConverter(ConverterConfig(formula_layout_fallback=True))
        conv.rapid._formula = object()  # formula engine present
        conv.rapid.recognize_formula = lambda p: r"\frac{x}{y}"
        img = Image.new("RGB", (100, 100))
        monkeypatch.setattr(conv, "_render_page_to_pil", lambda *a, **k: (img, 612.0, 792.0))
        monkeypatch.setattr(
            conv, "_layout_equation_boxes", lambda *a, **k: [(100.0, 400.0, 300.0, 420.0)]
        )
        md = conv._surgical_page_markdown(None, page, 0, tmp_path)
        assert r"\frac{x}{y}" in md

    def test_surgical_fallback_off_by_default(self, monkeypatch, tmp_path):
        """Without the flag, a no-math-font page skips formula recognition."""
        from bobine.converter import ConverterConfig, HybridConverter

        page = FakePage(text="some prose", chars=[])
        conv = HybridConverter(ConverterConfig())  # formula_layout_fallback=False
        conv.rapid._formula = object()
        called = []
        conv.rapid.recognize_formula = lambda p: called.append(p) or r"\frac{x}{y}"
        monkeypatch.setattr(
            conv, "_layout_equation_boxes", lambda *a, **k: [(100.0, 400.0, 300.0, 420.0)]
        )
        conv._surgical_page_markdown(None, page, 0, tmp_path)
        assert called == []  # recognizer never invoked

    def test_pdf_missing_raises_runtime_error(self, tmp_path):
        """convert_pdf fails loudly when it cannot open the PDF: a clear
        RuntimeError when pdf_oxide is absent, pdf_oxide's own error when the
        real backend is installed."""
        import importlib.util

        import pytest

        from bobine.pipeline import convert_to_markdown

        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        cfg = ConverterConfig(routing_mode=RoutingMode.NEVER)
        if importlib.util.find_spec("pdf_oxide") is None:
            # no backend → clear guidance error
            with pytest.raises(RuntimeError):
                convert_to_markdown(pdf, cfg, tmp_path / "work")
        else:
            # real backend → the invalid file surfaces pdf_oxide's own error
            with pytest.raises((OSError, RuntimeError)):
                convert_to_markdown(pdf, cfg, tmp_path / "work")
