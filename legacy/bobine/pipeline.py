"""Document ingestion orchestration (``bobine.pipeline``).

High-level, graph-agnostic pipeline shared by every entry point (CLI, API,
batch tools):

    document → markdown  (PDF via HybridConverter, Office via office_oxide,
                          text read as-is)
            → .md on disk
            → loose images moved into ``_assets/`` and links rewritten to
              ``okf-asset://<id>``
            → optional mordant lint
            → :class:`ConvertedDocument`

Nothing here touches a database or an embedding model — that stays in the
consumer (e.g. OKFgraph). The output contract is a directory of staged
markdown plus the staged asset store, ready to be handed to an import step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from bobine.assets import ASSET_STORE_DIRNAME, stage_images_as_okf_assets
from bobine.config import ConverterConfig
from bobine.converter import HybridConverter
from bobine.markdown import lint_markdown

# -- optional native deps (guarded, same contract as converter) --------------
try:
    from pdf_oxide import PdfDocument
except ImportError:  # pragma: no cover
    PdfDocument = None  # type: ignore[assignment]

try:
    from office_oxide import Document as OfficeDocument
except ImportError:  # pragma: no cover
    OfficeDocument = None  # type: ignore[assignment]


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"}
OFFICE_EXTS = SUPPORTED_EXTENSIONS - {".pdf"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".text"}

# Loose image files are assets, not documents — they are collected into the
# asset store but never converted themselves.
IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".svg",
    ".avif",
    ".heic",
    ".heif",
}


@dataclass
class ConvertedDocument:
    """Result of :func:`ingest_document` — a staged, linted markdown doc."""

    md_path: Path
    md_text: str
    image_dir: Path
    image_count: int = 0
    page_count: int = 0
    lint: dict[str, Any] = field(default_factory=dict)


def convert_to_markdown(
    path: str | Path,
    config: ConverterConfig,
    work_dir: str | Path,
    *,
    should_continue: Callable[[], bool] | None = None,
    on_page: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] = print,
) -> str:
    """Convert a single document to a markdown string.

    Dispatch by extension:

    - ``.pdf`` → :class:`HybridConverter` (pdf_oxide fast path + ONNX passes).
    - Office extensions → ``office_oxide`` ``to_markdown()``.
    - text extensions → read as UTF-8 (with ``errors="ignore"`` fallback).

    ``work_dir`` receives extracted images / crops. Returns the markdown text.
    Raises ``RuntimeError`` if a native backend is missing for the extension.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if ext == ".pdf":
        converter = HybridConverter(config, log=log)
        try:
            converter.ensure_models()
            return converter.convert_pdf(
                path=path,
                work_dir=work_dir,
                should_continue=should_continue or (lambda: True),
                on_page=on_page or (lambda idx, total: None),
            )
        finally:
            converter.close()

    if ext in OFFICE_EXTS:
        if OfficeDocument is None:
            raise RuntimeError(
                "office_oxide is not installed; cannot convert Office files. "
                "(pip install office_oxide)"
            )
        with OfficeDocument.open(str(path)) as doc:
            return doc.to_markdown()

    if ext in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(
        f"Unsupported extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS | TEXT_EXTS)}"
    )


def stage_images(
    md: str,
    source_path: str | Path,
    out_dir: str | Path,
    concept_stem: str,
) -> tuple[str, int]:
    """Collect loose images from ``out_dir`` and rewrite links to ``okf-asset://``.

    Mirrors the classic "converter dropped images in the work dir" flow:

    1. every loose image file sitting directly in ``out_dir`` is moved into
       ``out_dir/_assets/``;
    2. :func:`bobine.assets.stage_images_as_okf_assets` then rewrites
       ``![](local)`` links to ``![](okf-asset://<id>)`` and copies the bytes
       into the asset store (deduped, concept-scoped ids).

    Returns ``(rewritten_md, image_count)``.
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / ASSET_STORE_DIRNAME
    img_dir.mkdir(parents=True, exist_ok=True)
    for img in out_dir.glob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            try:
                img.rename(img_dir / img.name)
            except OSError:  # pragma: no cover — cross-device or locked file
                continue

    return stage_images_as_okf_assets(md, img_dir, Path(source_path), out_dir, concept_stem)


def ingest_document(
    path: str | Path,
    output_dir: str | Path,
    *,
    config: ConverterConfig | None = None,
    extract_images: bool | None = None,
    lint: bool = True,
    auto_fix: bool = True,
    should_continue: Callable[[], bool] | None = None,
    on_page: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] = print,
) -> ConvertedDocument:
    """Full ingestion pipeline for one document.

    Convert ``path`` to markdown (writing into ``output_dir``), move loose
    images into the asset store, rewrite links to ``okf-asset://``, and
    optionally lint the result with mordant.

    Returns a :class:`ConvertedDocument`. Nothing is imported anywhere — a
    consumer (e.g. an OKF graph router) receives the staged bundle.
    """
    path = Path(path)
    output_dir = Path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if config is None:
        config = ConverterConfig(extract_images=extract_images is not False)
    elif extract_images is not None:
        config = replace(config, extract_images=extract_images)

    output_dir.mkdir(parents=True, exist_ok=True)

    md = convert_to_markdown(
        path,
        config,
        output_dir,
        should_continue=should_continue,
        on_page=on_page,
        log=log,
    )

    stem = path.stem
    md_path = output_dir / f"{stem}.md"
    md_path.write_text(md, encoding="utf-8")

    # Stage images: move loose files, rewrite links, copy bytes into _assets.
    md_text, image_count = stage_images(md, path, output_dir, stem)
    md_path.write_text(md_text, encoding="utf-8")

    page_count = 0
    if path.suffix.lower() == ".pdf" and PdfDocument is not None:
        try:
            with PdfDocument(str(path)) as doc:
                try:
                    page_count = len(doc)
                except TypeError:
                    page_count = doc.page_count()
        except Exception:
            page_count = 0

    # Optional lint (fixes formatting issues in place).
    lint_result: dict[str, Any] = {}
    if lint:
        lint_result = lint_markdown(md_text, auto_fix=auto_fix)
        if lint_result["fixed"]:
            md_path.write_text(lint_result["content"], encoding="utf-8")

    return ConvertedDocument(
        md_path=md_path,
        md_text=md_path.read_text(encoding="utf-8"),
        image_dir=output_dir / ASSET_STORE_DIRNAME,
        image_count=image_count,
        page_count=page_count,
        lint=lint_result,
    )


def convert_directory(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    config: ConverterConfig | None = None,
    lint: bool = True,
    log: Callable[[str], None] = print,
) -> list[ConvertedDocument]:
    """Batch-convert every supported document in ``source_dir`` (recursive).

    Text files and PDF/Office files are converted into ``output_dir`` as
    staged markdown (images collected into ``_assets/``). Loose image files
    are skipped — they are assets, not documents. Returns the list of
    :class:`ConvertedDocument` results.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")

    results: list[ConvertedDocument] = []
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in IMAGE_EXTS:
            continue
        if ext not in (SUPPORTED_EXTENSIONS | TEXT_EXTS):
            continue
        try:
            doc = ingest_document(file_path, output_dir, config=config, lint=lint, log=log)
            results.append(doc)
            log(f"✓ {file_path.name} → {doc.md_path.name}")
        except Exception as e:
            log(f"✗ {file_path.name}: {e}")
    return results
