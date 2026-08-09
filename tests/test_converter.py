"""Tests for HybridConverter initialization and heuristics (ported from OKFgraph)."""

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
