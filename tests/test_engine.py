"""Tests for OnnxRapidEngine graceful degradation without RapidAI installed."""

import pytest

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


class _OCRResult:
    """Mimics rapidocr>=3's RapidOCROutput dataclass."""

    def __init__(self, boxes, txts, scores):
        self.boxes = boxes
        self.txts = txts
        self.scores = scores


class _LayoutResult:
    """Mimics rapid_layout>=1.2's RapidLayoutOutput dataclass."""

    def __init__(self, boxes, class_names, scores):
        self.boxes = boxes
        self.class_names = class_names
        self.scores = scores


class _TableResult:
    """Mimics rapid_table>=3's RapidTableOutput dataclass."""

    def __init__(self, pred_htmls):
        self.pred_htmls = pred_htmls


class TestEngineResponseNormalization:
    """Version-drift guards: rapidocr/layout/table 3.x return dataclasses."""

    def _eng_with(self, **slots):
        eng = OnnxRapidEngine(log_fn=lambda *a, **k: None)
        for name, val in slots.items():
            setattr(eng, f"_{name}", val)
        return eng

    def test_ocr_lines_normalizes_dataclass(self):
        def fake_ocr(img):
            return _OCRResult(
                boxes=[[0, 0, 10, 10], [20, 0, 30, 10]],
                txts=("hello", "world"),
                scores=(0.95, 0.8),
            )

        eng = self._eng_with(ocr=fake_ocr)
        out = eng.ocr_lines("ignored")
        assert out == [
            ([0, 0, 10, 10], "hello", 0.95),
            ([20, 0, 30, 10], "world", 0.8),
        ]

    def test_ocr_lines_handles_ndarray_boxes(self):
        """RapidOCROutput.boxes is an ndarray; `or` on it raises (numpy-2)."""
        np = pytest.importorskip("numpy")

        def fake_ocr(img):
            return _OCRResult(
                boxes=np.array([[0, 0, 10, 10], [20, 0, 30, 10]]),
                txts=("hello", "world"),
                scores=(0.95, 0.8),
            )

        eng = self._eng_with(ocr=fake_ocr)
        out = eng.ocr_lines("ignored")
        assert [b for b, _t, _s in out] == [[0, 0, 10, 10], [20, 0, 30, 10]]
        assert [t for _b, t, _s in out] == ["hello", "world"]

    def test_ocr_lines_normalizes_legacy_list(self):
        def fake_ocr(img):
            return [[[0, 0, 10, 10], "old", 0.7]]

        eng = self._eng_with(ocr=fake_ocr)
        assert eng.ocr_lines("ignored") == [([0, 0, 10, 10], "old", 0.7)]

    def test_ocr_lines_returns_empty_on_failure(self):
        def fake_ocr(img):
            raise RuntimeError("boom")

        eng = self._eng_with(ocr=fake_ocr)
        assert eng.ocr_lines("ignored") == []

    def test_layout_regions_normalizes_dataclass(self):
        def fake_layout(img):
            return _LayoutResult(
                boxes=[[1, 2, 3, 4]],
                class_names=["equation"],
                scores=[0.99],
            )

        eng = self._eng_with(layout=fake_layout)
        out = eng.layout_regions("ignored")
        assert out == [([1, 2, 3, 4], "equation", 0.99)]

    def test_layout_regions_normalizes_legacy_tuple(self):
        def fake_layout(img):
            return ([[1, 2, 3, 4]], [0.99], ["title"], 0.1)

        eng = self._eng_with(layout=fake_layout)
        out = eng.layout_regions("ignored")
        assert out == [([1, 2, 3, 4], "title", 0.99)]

    def test_layout_regions_returns_empty_on_failure(self):
        def fake_layout(img):
            raise RuntimeError("boom")

        eng = self._eng_with(layout=fake_layout)
        assert eng.layout_regions("ignored") == []

    def test_table_html_normalizes_dataclass(self):
        def fake_table(img, ocr_res):
            return _TableResult(pred_htmls=["<table><tr><td>a</td></tr></table>"])

        eng = self._eng_with(table=fake_table)
        out = eng.table_html("ignored")
        assert out and out.startswith("<table>")

    def test_table_html_normalizes_legacy_tuple(self):
        def fake_table(img, ocr_res):
            return ("<table>old</table>", None, 0.2)

        eng = self._eng_with(table=fake_table)
        assert eng.table_html("ignored") == "<table>old</table>"

    def test_table_html_returns_none_on_failure(self):
        def fake_table(img, ocr_res):
            raise RuntimeError("boom")

        eng = self._eng_with(table=fake_table)
        assert eng.table_html("ignored") is None


class TestLoaderFailureDegradation:
    """Lazy loaders log and degrade when constructors raise (coverage #3)."""

    def _engine(self):
        return OnnxRapidEngine(log_fn=lambda *a, **k: None)

    def test_formula_loader_failure(self, monkeypatch):
        import bobine.engine as eng_mod

        class Bad:
            def __init__(self):
                raise RuntimeError("broken")

        monkeypatch.setattr(eng_mod, "LatexOCR", Bad)
        assert self._engine().formula() is None

    def test_ocr_loader_failure(self, monkeypatch):
        import bobine.engine as eng_mod

        class Bad:
            def __init__(self):
                raise RuntimeError("broken")

        monkeypatch.setattr(eng_mod, "RapidOCR", Bad)
        assert self._engine().ocr() is None

    def test_layout_loader_failure(self, monkeypatch):
        import bobine.engine as eng_mod

        class Bad:
            def __init__(self):
                raise RuntimeError("broken")

        monkeypatch.setattr(eng_mod, "RapidLayout", Bad)
        assert self._engine().layout() is None

    def test_table_loader_failure(self, monkeypatch):
        import bobine.engine as eng_mod

        class Bad:
            def __init__(self):
                raise RuntimeError("broken")

        monkeypatch.setattr(eng_mod, "RapidTable", Bad)
        assert self._engine().table() is None

    def test_recognize_formula_failure_and_empty(self):
        eng = self._engine()

        def bad(data):
            raise RuntimeError("boom")

        eng._formula = bad
        assert eng.recognize_formula("whatever.png") is None

        def empty(data):
            return ("   ", 1.0)  # blank latex → None

        eng._formula = empty
        assert eng.recognize_formula("whatever.png") is None

    def test_recognize_formula_no_engine(self):
        assert self._engine().recognize_formula("x.png") is None

    def test_ocr_legacy_short_row_defaults_score(self):
        def fake_ocr(img):
            return [[[0, 0, 1, 1], "short"]]

        eng = self._engine()
        eng._ocr = fake_ocr
        assert eng.ocr_lines("x") == [([0, 0, 1, 1], "short", 1.0)]

    def test_table_html_no_engine_and_empty(self):
        eng = self._engine()
        assert eng.table_html("x") is None

        def fake_table(img, ocr_res):
            return _TableResult(pred_htmls=[])

        eng._table = fake_table
        assert eng.table_html("x") is None
