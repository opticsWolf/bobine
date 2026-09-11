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

On import, bobine points ``ORT_DYLIB_PATH`` at the pip-installed
``onnxruntime`` library automatically (an already-set ``ORT_DYLIB_PATH``
always wins). Without any runtime the fast Office/text paths still work.
"""

import os as _os


def _ensure_ort_dylib() -> str | None:
    """Point ORT_DYLIB_PATH at a pip-installed onnxruntime, if needed.

    Returns the resolved path, or None when no runtime was found (the
    caller keeps working — only ONNX-backed pages need it). Runs before
    the native module is imported; ort loads lazily on first session, so
    setting the env var here is early enough.
    """
    if _os.environ.get("ORT_DYLIB_PATH"):
        return _os.environ["ORT_DYLIB_PATH"]
    try:
        import onnxruntime as _ort  # also provided by onnxruntime-gpu
    except ImportError:
        return None
    _warm_gpu_dlls(_ort)
    capi = _os.path.join(_os.path.dirname(_ort.__file__), "capi")
    candidates: list[str] = []
    if _os.name == "nt":
        candidates = ["onnxruntime.dll"]
    elif _os.name == "posix":
        import glob as _glob

        if __import__("sys").platform == "darwin":
            candidates = ["libonnxruntime.dylib"]
        else:
            # Prefer the versioned .so (libonnxruntime.so may be absent).
            found = sorted(_glob.glob(_os.path.join(capi, "libonnxruntime.so.*")))
            candidates = [ _os.path.basename(p) for p in found ] + ["libonnxruntime.so"]
    for lib in candidates:
        path = _os.path.join(capi, lib)
        if _os.path.isfile(path):
            _os.environ["ORT_DYLIB_PATH"] = path
            if _os.name == "nt":
                # Let the GPU build resolve its CUDA DLL siblings.
                _os.add_dll_directory(capi)
            return path
    return None


def _warm_gpu_dlls(_ort) -> None:
    """Preload NVIDIA DLLs for GPU builds so direct loads resolve.

    Only for GPU builds (CUDA EP registered): CPU-only packages skip
    silently. `import onnxruntime` alone does NOT load them — its
    `preload_dlls()` helper runs only on explicit call — but ort loads
    `onnxruntime.dll` directly (bypassing it), so without this the CUDA
    EP fails at the first Conv node with `cudnn64_9.dll` not found and
    every page silently falls back. Preloaded DLLs are process-global,
    so ort's later LoadLibrary calls resolve. Never breaks import.
    """
    try:
        providers = _ort.get_available_providers()
    except Exception:
        return
    if "CUDAExecutionProvider" not in providers:
        return
    try:
        _ort.preload_dlls()
    except Exception:
        pass


ORT_DYLIB_PATH = _ensure_ort_dylib()

from bobine._native import (
    ConverterConfig,
    HybridConverter,
    RoutingMode,
    FormulaBackend,
    ModelPrecision,
    convert_to_markdown,
    ConvertedDocument,
    ExcelDocument,
    convert_excel,
    ingest_document,
    convert_directory,
)

__all__ = [
    "ConverterConfig",
    "HybridConverter",
    "RoutingMode",
    "FormulaBackend",
    "ModelPrecision",
    "convert_to_markdown",
    "ConvertedDocument",
    "ExcelDocument",
    "convert_excel",
    "ingest_document",
    "convert_directory",
    "ORT_DYLIB_PATH",
]
