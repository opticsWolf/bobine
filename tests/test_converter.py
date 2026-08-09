"""Tests for HybridConverter initialization and heuristics (ported from OKFgraph)."""

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

    def test_pdf_missing_raises_runtime_error(self, tmp_path):
        """Without pdf_oxide installed, convert_pdf raises a clear error."""
        import pytest

        from bobine.pipeline import convert_to_markdown

        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        cfg = ConverterConfig(routing_mode=RoutingMode.NEVER)
        # convert_to_markdown raises RuntimeError when pdf_oxide is absent
        with pytest.raises(RuntimeError):
            convert_to_markdown(pdf, cfg, tmp_path / "work")
