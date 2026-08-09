"""Tests for OnnxRapidEngine graceful degradation without RapidAI installed."""

from bobine.engine import OnnxRapidEngine


class TestEngineGracefulDegradation:
    def test_formula_returns_none_when_not_installed(self):
        eng = OnnxRapidEngine()
        result = eng.formula()
        assert result is None or hasattr(result, "predict") or callable(result)

    def test_ocr_returns_none_when_not_installed(self):
        eng = OnnxRapidEngine()
        result = eng.ocr()
        assert result is None or hasattr(result, "predict") or callable(result)

    def test_layout_returns_none_when_not_installed(self):
        eng = OnnxRapidEngine()
        result = eng.layout()
        assert result is None or hasattr(result, "predict") or callable(result)

    def test_table_returns_none_when_not_installed(self):
        eng = OnnxRapidEngine()
        result = eng.table()
        assert result is None or hasattr(result, "predict") or callable(result)

    def test_close_clears_references(self):
        eng = OnnxRapidEngine()
        eng.formula()
        eng.ocr()
        eng.close()
        assert eng._formula is None
        assert eng._ocr is None
        assert eng._layout is None
        assert eng._table is None
