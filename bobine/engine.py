"""OnnxRapidEngine — all ONNX heavy passes behind lazy loaders (package ``bobine``).

Wraps the vendored RapidLaTeXOCR and the RapidAI family packages (rapidocr,
rapid_layout, rapid_table) so models are loaded on first use. A clean
born-digital paper loads only the tiny formula model, never the OCR/layout/
table stack.

Version-sensitive RapidAI calls are flagged ``# VERIFY``. Verified against
rapidocr 3.9.2 / rapid_layout 1.2.1 / rapid_table 3.0.2 (2026-08-09): the 3.x
line returns dataclasses (RapidOCROutput / RapidLayoutOutput /
RapidTableOutput) instead of tuples, and RapidOCR() takes no kwargs — the
normalization branches below handle both shapes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

# numpy is optional (needed only to adapt OCR boxes for rapid_table >=3)
try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

# -- lazy imports: all guarded so the module loads without RapidAI installed --

try:
    from bobine._vendor.rapid_latex_ocr import LatexOCR
except ImportError:  # pragma: no cover — optional deps (onnxruntime/tokenizers/cv2) missing
    LatexOCR = None  # type: ignore[assignment]

try:
    from rapidocr import RapidOCR  # VERIFY: older wheels: rapidocr_onnxruntime
except ImportError:  # pragma: no cover
    RapidOCR = None  # type: ignore[assignment]

try:
    from rapid_layout import RapidLayout
except ImportError:  # pragma: no cover
    RapidLayout = None  # type: ignore[assignment]

try:
    from rapid_table import RapidTable
except ImportError:  # pragma: no cover
    RapidTable = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


class OnnxRapidEngine:
    """All ONNX heavy passes behind lazy loaders. No PaddlePaddle."""

    def __init__(
        self,
        log_fn: Callable[[str], None] = print,
        ort_providers: list[str] | None = None,
    ) -> None:
        self.log = log_fn
        self.ort_providers = ort_providers
        self._formula: object | None = None
        self._ocr: object | None = None
        self._layout: object | None = None
        self._table: object | None = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def formula(self) -> object | None:
        """Return the RapidLaTeXOCR instance (lazy)."""
        if self._formula is None:
            if LatexOCR is None:
                self.log(
                    "⚠️  formula OCR unavailable (pip install bobine[formula]); math stays as text."
                )
                return None
            self.log("⚙️  Loading formula OCR (vendored RapidLaTeXOCR → LaTeX)…")
            try:
                self._formula = LatexOCR()  # VERIFY: pass model paths for offline
            except Exception as e:
                self.log(f"⚠️  Could not load RapidLaTeXOCR: {e}")
                return None
        return self._formula

    def ocr(self) -> object | None:
        """Return the RapidOCR instance (lazy)."""
        if self._ocr is None and RapidOCR is not None:
            self.log("⚙️  Loading RapidOCR (text det+rec)…")
            try:
                self._ocr = RapidOCR()  # rapidocr>=3: defaults = onnxruntime/CPU, use_cls=True
            except Exception as e:
                self.log(f"⚠️  Could not load RapidOCR: {e}")
                return None
        return self._ocr

    def layout(self) -> object | None:
        """Return the RapidLayout instance (lazy)."""
        if self._layout is None and RapidLayout is not None:
            self.log("⚙️  Loading RapidLayout…")
            try:
                self._layout = RapidLayout()
            except Exception as e:
                self.log(f"⚠️  Could not load RapidLayout: {e}")
                return None
        return self._layout

    def table(self) -> object | None:
        """Return the RapidTable instance (lazy)."""
        if self._table is None and RapidTable is not None:
            self.log("⚙️  Loading RapidTable…")
            try:
                self._table = RapidTable()
            except Exception as e:
                self.log(f"⚠️  Could not load RapidTable: {e}")
                return None
        return self._table

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all model references."""
        self._formula = None
        self._ocr = None
        self._layout = None
        self._table = None

    # ------------------------------------------------------------------
    # Formula: image bytes / path → LaTeX
    # ------------------------------------------------------------------

    def recognize_formula(self, crop_path: str) -> str | None:
        """Run a single crop through RapidLaTeXOCR and return LaTeX, or None."""
        eng = self.formula()
        if eng is None:
            return None
        try:
            data = Path(crop_path).read_bytes()
            res, _elapse = eng(data)  # VERIFY: returns (latex_str, elapse)
            return (res or "").strip() or None
        except Exception as e:
            self.log(f"   ⚠️  formula recog failed: {e}")
            return None

    # ------------------------------------------------------------------
    # OCR: image → [(box, text, score)]
    # ------------------------------------------------------------------

    def ocr_lines(self, img) -> list[tuple]:
        """Run RapidOCR on ``img`` (path, ndarray, or bytes).

        Returns a list of ``(box, text, score)`` tuples.
        """
        eng = self.ocr()
        if eng is None:
            return []
        try:
            result = eng(img)
        except Exception as e:
            self.log(f"   ⚠️  OCR failed: {e}")
            return []

        # Normalize across versions:
        #   v3 -> result.boxes / .txts / .scores
        #   older -> list of [box, text, score]
        if hasattr(result, "txts"):  # VERIFY
            boxes = getattr(result, "boxes", None)
            if hasattr(boxes, "tolist"):  # RapidOCROutput.boxes is an ndarray
                boxes = boxes.tolist()
            boxes = boxes or []
            txts = getattr(result, "txts", None) or []
            scores = getattr(result, "scores", None) or []
            return list(zip(boxes, txts, scores, strict=False))
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)):
            return [(r[0], r[1], r[2] if len(r) > 2 else 1.0) for r in result]
        return []

    # ------------------------------------------------------------------
    # Layout: image → [(box, label, score)]
    # ------------------------------------------------------------------

    def layout_regions(self, img) -> list[tuple]:
        """Run RapidLayout on ``img``.

        Returns a list of ``(box, label, score)`` tuples.
        """
        eng = self.layout()
        if eng is None:
            return []
        try:
            out = eng(img)  # VERIFY: v1.2+ → RapidLayoutOutput dataclass; older → 4-tuple
            if hasattr(out, "boxes"):
                boxes = getattr(out, "boxes", None) or []
                labels = getattr(out, "class_names", None) or []
                scores = getattr(out, "scores", None) or []
                return list(zip(boxes, labels, scores, strict=False))
            boxes, scores, labels, _elapse = out
            return list(zip(boxes, labels, scores, strict=False))
        except Exception as e:
            self.log(f"   ⚠️  layout failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Table: region crop (+ OCR) → HTML
    # ------------------------------------------------------------------

    def table_html(self, crop_img) -> str | None:
        """Run RapidTable on ``crop_img`` (with OCR text fill).

        Returns HTML table string, or None on failure.
        """
        eng = self.table()
        if eng is None:
            return None
        try:
            lines = self.ocr_lines(crop_img)
            # rapid_table>=3 expects per-image [boxes_array, txts_tuple, scores_tuple]
            if lines and _np is not None:
                boxes = _np.array([b for b, _t, _s in lines], dtype=_np.float32)
                ocr_res = [
                    [boxes, tuple(t for _b, t, _s in lines), tuple(float(s) for _b, _t, s in lines)]
                ]
            else:
                ocr_res = None
            out = eng(crop_img, ocr_res)  # VERIFY: v3 → RapidTableOutput.pred_htmls; older → tuple
            if hasattr(out, "pred_htmls"):
                htmls = getattr(out, "pred_htmls", None) or []
                html = htmls[0] if htmls else None
            elif isinstance(out, tuple):
                html = out[0]
            else:
                html = getattr(out, "pred_html", None) or getattr(out, "html", None)
            return html
        except Exception as e:
            self.log(f"   ⚠️  table recog failed: {e}")
            return None
