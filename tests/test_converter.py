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
