"""bobine — Fast PDF/Office/Text → Markdown ingestion engine.

Powered by pdf_oxide (Rust) + ONNX Runtime for formula OCR.
Zero Python ML dependencies (no torch, no optimum, no opencv).

Usage::

    import bobine

    config = bobine.ConverterConfig(
        routing_mode=bobine.RoutingMode.Surgical,
        formula_dpi=200,
    )
    converter = bobine.HybridConverter(config, cache_dir="~/.cache/bobine")
    md = converter.convert_pdf("paper.pdf", work_dir="/tmp/out")

Heavy (ONNX) modes need an ONNX Runtime library::

    pip install bobine[cpu]   # or bobine[gpu] for CUDA

On import, bobine finds that library with embroider's locator — the same
one every embroider model open and okfgraph use — and points
``ORT_DYLIB_PATH`` at it (an already-set ``ORT_DYLIB_PATH`` always wins).
The locator reads the installed ``onnxruntime`` dist's metadata and never
imports the module, so a host process keeps one runtime instead of two.
Without any runtime the fast Office/text paths still work; only ONNX-backed
pages need it. :func:`runtime_info` reports what was found.
"""

import sys as _sys

# Importing the native module does not load ONNX Runtime: `ort` is built
# with `load-dynamic` and resolves the library when the first session
# builds. So the locator can run after this import, which is what lets it
# be the same code embroider and okfgraph run.
from . import _native

#: What bobine decided about ONNX Runtime at import: the library `ort`
#: will load, where it came from, and the CUDA folders put on `PATH`.
_RUNTIME = dict(_native.bootstrap_runtime(_sys.path))

#: The ONNX Runtime library bobine will load, or ``None`` when nothing was
#: found (only ONNX-backed pages need it). An already-set
#: ``ORT_DYLIB_PATH`` always wins.
ORT_DYLIB_PATH = _RUNTIME["dylib"]


def runtime_info() -> dict:
    """What bobine decided about ONNX Runtime, without importing it.

    ``{"dylib", "source", "version", "set_by", "cuda_dirs"}``:

    - ``dylib`` — the library ``ort`` will load, or ``None``.
    - ``source`` — ``"env"`` (an explicit ``ORT_DYLIB_PATH`` won),
      ``"pip:<distribution>"`` (a wheel was found), or ``"os"`` (nothing
      found: ``ort`` would try the OS loader path, which can hold a stale
      copy — ONNX-backed pages should refuse to open).
    - ``version`` — that distribution's version.
    - ``set_by`` — who chose it: ``"env"``, ``"bobine"``, or ``"none"``.
    - ``cuda_dirs`` — CUDA folders this process prepended to ``PATH``
      before the first session.

    The search runs once, at import (see :func:`bobine._native.bootstrap_runtime`
    and ``src/runtime.rs``); this returns that decision. For a full
    inventory — every dist, provider DLLs, the CUDA probe — use
    ``embroider.runtime_info()``.
    """
    return dict(_native.runtime_info())


from bobine._native import (
    ConverterConfig,
    HybridConverter,
    RoutingMode,
    ImageMode,
    FormulaBackend,
    TableBackend,
    ModelPrecision,
    convert_to_markdown,
    ConvertedDocument,
    ExcelDocument,
    convert_excel,
    convert_jats,
    JatsDocument,
    ingest_document,
    convert_directory,
    model_status,
)

__all__ = [
    "ConverterConfig",
    "HybridConverter",
    "RoutingMode",
    "ImageMode",
    "FormulaBackend",
    "TableBackend",
    "ModelPrecision",
    "convert_to_markdown",
    "ConvertedDocument",
    "ExcelDocument",
    "convert_excel",
    "convert_jats",
    "JatsDocument",
    "ingest_document",
    "convert_directory",
    "model_status",
    "ORT_DYLIB_PATH",
    "runtime_info",
]
