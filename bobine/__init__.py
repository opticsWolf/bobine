"""bobine — PDF / Office / text → Markdown ingestion engine.

Standalone module extracted from the OKFgraph project. Runs on a single
``onnxruntime`` wheel with no CUDA-version coupling (RapidAI family + pdf_oxide),
and needs nothing but the stdlib for plain text / markdown documents.

Public API
----------
Conversion (PDF/Office)
    - ``ConverterConfig`` / ``RoutingMode`` — configuration and routing modes.
    - ``HybridConverter`` — core conversion pipeline (Qt-independent).
    - ``OnnxRapidEngine`` — lazy ONNX model manager.
    - ``html_tables_to_gfm`` — HTML table → GFM pipe-table converter.

Images
    - ``stage_images_as_okf_assets`` — okf-asset:// staging for extracted images.
    - ``asset_id`` / ``ASSET_STORE_DIRNAME`` — asset naming convention.

Text-type documents
    - ``Document`` — normalized document model (id, title, body, tags, …).
    - ``load_markdown_document`` — frontmatter-aware .md loading.
    - ``wrap_thoughts`` — raw reasoning text → OKF-compliant markdown.
    - ``lint_markdown`` / ``lint_markdown_file`` — mordant linting (guarded).

Orchestration
    - ``convert_to_markdown`` — dispatch PDF / Office / text → markdown string.
    - ``stage_images`` — collect extracted images into an asset store.
    - ``ingest_document`` — full pipeline: convert → write .md → stage assets
      → lint, returns a ``ConvertedDocument``.

Versioning
    - ``check_rapid_versions`` — runtime version check for RapidAI packages.
"""

from bobine.assets import (
    ASSET_STORE_DIRNAME,
    asset_id,
    stage_images_as_okf_assets,
)
from bobine.config import ConverterConfig, RoutingMode
from bobine.converter import HybridConverter
from bobine.documents import Document, load_markdown_document, wrap_thoughts
from bobine.engine import OnnxRapidEngine
from bobine.markdown import lint_markdown, lint_markdown_file
from bobine.pipeline import (
    OFFICE_EXTS,
    SUPPORTED_EXTENSIONS,
    TEXT_EXTS,
    ConvertedDocument,
    convert_directory,
    convert_to_markdown,
    ingest_document,
    stage_images,
)
from bobine.tables import html_tables_to_gfm
from bobine.versions import check_rapid_versions

__version__ = "0.1.0"

__all__ = [
    "ASSET_STORE_DIRNAME",
    "OFFICE_EXTS",
    "SUPPORTED_EXTENSIONS",
    "TEXT_EXTS",
    "ConvertedDocument",
    "ConverterConfig",
    "Document",
    "HybridConverter",
    "OnnxRapidEngine",
    "RoutingMode",
    "__version__",
    "asset_id",
    "check_rapid_versions",
    "convert_directory",
    "convert_to_markdown",
    "html_tables_to_gfm",
    "ingest_document",
    "lint_markdown",
    "lint_markdown_file",
    "load_markdown_document",
    "stage_images",
    "stage_images_as_okf_assets",
    "wrap_thoughts",
]
