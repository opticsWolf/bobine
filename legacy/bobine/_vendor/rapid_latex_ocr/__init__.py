# -*- encoding: utf-8 -*-
# @Author: SWHL (vendored into bobine, MIT licensed)
from .main import LaTeXOCR

# Legacy alias: older builds shipped the class as `LatexOCR`; keep both so
# bobine's engine import works regardless of which build was vendored.
LatexOCR = LaTeXOCR

__all__ = ["LaTeXOCR", "LatexOCR"]
