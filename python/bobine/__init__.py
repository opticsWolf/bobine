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
"""

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
]
