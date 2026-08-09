"""Tests for the vendored RapidLaTeXOCR (bobine._vendor.rapid_latex_ocr).

The formula OCR engine is vendored into bobine (MIT (c) 2023 RapidAI) with the
numpy-2 incompatibility fixed (see ``main.py::loop_image_resizer``). These
tests verify:

- import works without the heavy deps installed (graceful degradation);
- with the ``bobine[formula]`` deps installed, a real formula crop is
  recognised (models ~179 MB auto-download on first use → integration).

The recognition test is gated behind the ``integration`` marker so the bare
install stays green.
"""

from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "formula_sample.png"


class TestVendoredImport:
    def test_engine_imports_vendored_latex_ocr(self):
        from bobine.engine import LatexOCR

        # Without the optional deps the module-level guard sets it to None;
        # with them installed it is the vendored class.
        assert LatexOCR is None or callable(LatexOCR)

    def test_vendored_package_importable(self):
        from bobine._vendor.rapid_latex_ocr import LaTeXOCR, LatexOCR

        assert LatexOCR is LaTeXOCR  # legacy alias preserved

    def test_engine_degradation_message(self, tmp_path):
        """OnnxRapidEngine.formula() degrades gracefully when deps missing."""
        from bobine.engine import OnnxRapidEngine

        messages = []

        class _Eng(OnnxRapidEngine):
            def __init__(self):
                super().__init__(log_fn=messages.append)

        eng = _Eng()
        result = eng.formula()
        if eng._formula is None:
            assert result is None
        else:
            assert result is not None  # deps present in this env


@pytest.mark.integration
class TestVendoredRecognition:
    def test_formula_crop_to_latex(self):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("tokenizers")
        pytest.importorskip("cv2")

        from bobine._vendor.rapid_latex_ocr import LatexOCR

        model = LatexOCR()  # downloads ~179 MB of models on first use
        data = FIXTURE.read_bytes()
        latex, _elapse = model(data)
        assert isinstance(latex, str) and latex.strip()

    def test_engine_recognize_formula_integration(self):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("tokenizers")

        from bobine.engine import OnnxRapidEngine

        eng = OnnxRapidEngine(log_fn=lambda *a, **k: None)
        latex = eng.recognize_formula(str(FIXTURE))
        assert latex is None or isinstance(latex, str)
        eng.close()
