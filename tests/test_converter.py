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


class TestCoverageHelpers:
    """Guard/fallback branches driven with fakes (coverage #3)."""

    def test_page_math_signal_from_spans(self):
        from bobine.converter import HybridConverter

        class Span:
            def __init__(self, text, font_name):
                self.text = text
                self.font_name = font_name

        page = type(
            "P", (), {"text": "", "spans": [Span("ab", "CMMI10"), Span("cd", "Helvetica")]}
        )()
        conv = HybridConverter(ConverterConfig())
        math, total = conv._page_math_signal(page)
        assert math == 2 and total == 0

    def test_is_scanned_false_when_chars_raise(self):
        from bobine.converter import HybridConverter

        class BadPage:
            @property
            def chars(self):
                raise RuntimeError("boom")

        assert HybridConverter(ConverterConfig())._is_scanned(BadPage()) is False

    def test_needs_paddle_exception_logs_and_falls_back(self):
        from bobine.converter import HybridConverter

        logs = []
        conv = HybridConverter(ConverterConfig(routing_mode=RoutingMode.AUTO), log=logs.append)

        class BadPage:
            @property
            def chars(self):
                raise RuntimeError("boom")

        assert conv._needs_paddle(BadPage()) is False
        assert any("routing heuristic failed" in m for m in logs)

    def test_needs_paddle_auto_math_signal(self):
        from bobine.converter import HybridConverter

        # text-layer math signal exceeds threshold → full pipeline in AUTO
        page = FakePage(text="∑" * 40, chars=[], width=612.0, height=792.0)
        conv = HybridConverter(
            ConverterConfig(routing_mode=RoutingMode.AUTO, math_char_threshold=30)
        )
        assert conv._needs_paddle(page) is True

    def test_crop_pil_degenerate_box(self):
        from bobine.converter import HybridConverter

        conv = HybridConverter(ConverterConfig())
        assert conv._crop_pil(None, (0, 0, 1, 1), 100.0, 150) is None

    def test_ws_replace_no_match_and_tolerant(self):
        from bobine.converter import HybridConverter

        conv = HybridConverter(ConverterConfig())
        assert conv._ws_replace("hello world here", "nope nope", "X") is None
        out = conv._ws_replace("hello   world\nhere", "hello world", "X")
        assert out == "X\nhere"

    def test_obj_to_pil_variants(self, monkeypatch):
        from PIL import Image

        import bobine.converter as conv_mod
        from bobine.converter import HybridConverter

        conv = HybridConverter(ConverterConfig())
        img = Image.new("RGB", (4, 4))
        assert conv._obj_to_pil(img) is img  # PIL passthrough
        import io as _io

        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        assert conv._obj_to_pil({"data": buf.getvalue()}) is not None  # dict
        assert conv._obj_to_pil(b"\x89PNG") is None  # corrupt bytes
        monkeypatch.setattr(conv_mod, "PILImage", None)
        assert conv._obj_to_pil(img) is None  # PIL unavailable

    def test_render_page_to_png_variants(self, tmp_path, monkeypatch):
        from PIL import Image

        import bobine.converter as conv_mod
        from bobine.converter import HybridConverter

        conv = HybridConverter(ConverterConfig())
        out = tmp_path / "p.png"
        assert conv._render_page_to_png(None, None, 0, out, dpi=72) is False  # no obj
        conv._render_raw = lambda *a, **k: {"data": b"not-an-image"}
        assert conv._render_page_to_png(None, None, 0, out, dpi=72) is True  # dict bytes written
        img = Image.new("RGB", (4, 4))
        conv._render_raw = lambda *a, **k: img
        assert conv._render_page_to_png(None, None, 0, out, dpi=72) is True  # PIL save
        # ndarray obj + PIL unavailable → False (numpy-optional branch)
        np_mod = pytest.importorskip("numpy")
        monkeypatch.setattr(conv_mod, "PILImage", None)
        conv._render_raw = lambda *a, **k: np_mod.zeros((4, 4), dtype=np_mod.uint8)
        assert conv._render_page_to_png(None, None, 0, out, dpi=72) is False  # PIL gone

    def test_extract_page_images_variants(self, tmp_path):
        from bobine.converter import HybridConverter

        conv = HybridConverter(ConverterConfig())

        class AttrImage:
            data = b"\x89PNG\r\n\x1a\nrest"
            format = "png"

        class NoData:
            pass

        # dict-style + attr-style + no-data skip
        doc = type(
            "D",
            (),
            {
                "extract_image_bytes": lambda self, i: [
                    {"data": b"\x89PNG\r\n\x1a\nx", "format": ".PNG"},
                    NoData(),
                    AttrImage(),
                ]
            },
        )()
        paths = conv._extract_page_images(doc, None, 0, tmp_path)
        assert len(paths) == 2  # dict + attr images written, NoData skipped
        assert all(p.exists() for p in paths)

        # backend raises → logged, empty list
        logs = []
        conv.log = logs.append
        bad = type(
            "B",
            (),
            {"extract_image_bytes": lambda self, i: (_ for _ in ()).throw(RuntimeError("x"))},
        )()
        assert conv._extract_page_images(bad, None, 0, tmp_path) == []
        assert any("image extraction failed" in m for m in logs)

    def test_layout_equation_boxes_np_missing(self, monkeypatch):
        import bobine.converter as conv_mod
        from bobine.converter import HybridConverter

        monkeypatch.setattr(conv_mod, "np", None)
        conv = HybridConverter(ConverterConfig(formula_layout_fallback=True))
        assert conv._layout_equation_boxes(None, None, 0) == []

    def test_layout_equation_boxes_render_missing(self, monkeypatch):
        from bobine.converter import HybridConverter

        conv = HybridConverter(ConverterConfig(formula_layout_fallback=True))
        monkeypatch.setattr(conv, "_render_page_to_pil", lambda *a, **k: None)
        assert conv._layout_equation_boxes(None, None, 0) == []

    def test_full_structure_render_missing(self, tmp_path):
        from bobine.converter import HybridConverter

        conv = HybridConverter(ConverterConfig(routing_mode=RoutingMode.ALWAYS))
        monkeypatch = __import__("pytest").MonkeyPatch()
        monkeypatch.setattr(conv, "_render_page_to_pil", lambda *a, **k: None)
        try:
            assert conv._full_structure_page_markdown(None, FakePage(), 0, tmp_path) is None
        finally:
            monkeypatch.undo()

    def test_full_structure_np_missing(self, tmp_path, monkeypatch):
        from PIL import Image

        import bobine.converter as conv_mod
        from bobine.converter import HybridConverter

        logs = []
        conv = HybridConverter(ConverterConfig(routing_mode=RoutingMode.ALWAYS), log=logs.append)
        img = Image.new("RGB", (100, 100))
        monkeypatch.setattr(conv, "_render_page_to_pil", lambda *a, **k: (img, 612.0, 792.0))
        monkeypatch.setattr(conv_mod, "np", None)
        assert conv._full_structure_page_markdown(None, FakePage(), 0, tmp_path) is None
        assert any("numpy not available" in m for m in logs)

    def test_surgical_render_unavailable(self, tmp_path):
        from bobine.converter import HybridConverter

        logs = []
        conv = HybridConverter(ConverterConfig(routing_mode=RoutingMode.SURGICAL), log=logs.append)
        conv.rapid._formula = object()
        page = FakePage(
            text="x",
            chars=[FakeChar("x", font_name="CMMI10", bbox=(100.0, 700.0, 108.0, 712.0))] * 6,
        )
        monkeypatch = __import__("pytest").MonkeyPatch()
        monkeypatch.setattr(conv, "_render_page_to_pil", lambda *a, **k: None)
        try:
            md = conv._surgical_page_markdown(None, page, 0, tmp_path)
            assert md == "x"
            assert any("render unavailable" in m for m in logs)
        finally:
            monkeypatch.undo()

    def test_convert_office_missing_backend(self, monkeypatch):
        import bobine.converter as conv_mod
        from bobine.converter import HybridConverter

        monkeypatch.setattr(conv_mod, "OfficeDocument", None)
        with pytest.raises(RuntimeError):
            HybridConverter(ConverterConfig()).convert_office("x.docx")

    def test_convert_office_success(self, monkeypatch):
        import bobine.converter as conv_mod
        from bobine.converter import HybridConverter

        class FakeOffice:
            def __init__(self, path):
                pass

            @classmethod
            def open(cls, path):
                return cls(path)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def to_markdown(self):
                return "# from office"

        monkeypatch.setattr(conv_mod, "OfficeDocument", FakeOffice)
        assert HybridConverter(ConverterConfig()).convert_office("x.docx") == "# from office"
        assert HybridConverter(ConverterConfig()).convert_office("x.docx") == "# from office"
