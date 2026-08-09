# bobine

Standalone **PDF / Office / text → Markdown ingestion engine**, extracted from
the OKFgraph project as a self-contained module. Runs on a single
`onnxruntime` wheel with no CUDA-version coupling — RapidAI family + `pdf_oxide`
for PDFs, pure Python for text documents.

Dual-licensed under the terms of either the MIT License or the Apache License,
Version 2.0 — you may choose either (see [LICENSE](LICENSE)).

## Why bobine?

The ingestion pipeline was entangled with the knowledge-graph project it served.
`bobine` moves the whole pipeline — conversion, image staging, markdown
linting, document normalization — into its own package so any consumer (a
graph, a CLI, an MCP server, a batch tool) can reuse it without importing a
database stack.

## Installation

```bash
pip install -e .                # core (Pillow only)
pip install -e ".[pdf-ingest]"  # + pdf_oxide + RapidAI ONNX passes
pip install -e ".[markdown]"    # + mordant linting + frontmatter parsing
```

Everything is optional: the package imports with zero dependencies and
degrades gracefully (no-op fast paths, clear `RuntimeError`s when a backend
is missing).

## Quick start

```python
from bobine import ConverterConfig, RoutingMode, ingest_document

# One PDF → staged, linted markdown in ./out (images → ./out/_assets,
# links rewritten to okf-asset://<id>)
result = ingest_document(
    "paper.pdf",
    "out",
    config=ConverterConfig(
        routing_mode=RoutingMode.SURGICAL,
    ),
)
print(result.md_path, result.image_count, result.page_count)

# Text documents need no native deps at all
doc = ingest_document("notes.txt", "out")
```

### PDF conversion with ONNX heavy passes

`HybridConverter` routes pages through four modes:

| Mode      | Behaviour                                                                 |
|-----------|---------------------------------------------------------------------------|
| `NEVER`   | Fast path only (pdf_oxide). No ONNX models loaded.                        |
| `AUTO`    | Heuristics per page → full ONNX layout + OCR on flagged pages.            |
| `SURGICAL`| Formula crops via RapidLaTeXOCR only; full pipeline just for scans.       |
| `ALWAYS`  | Every page through the full ONNX layout + OCR pipeline.                   |

```python
from bobine import HybridConverter, ConverterConfig, RoutingMode

conv = HybridConverter(ConverterConfig(routing_mode=RoutingMode.AUTO))
conv.ensure_models()
md = conv.convert_pdf(
    "paper.pdf", work_dir="work", should_continue=lambda: True, on_page=lambda i, n: None
)
conv.close()
```

### Text-type documents

```python
from bobine import load_markdown_document, wrap_thoughts, lint_markdown

doc = load_markdown_document("note.md")  # frontmatter-aware
thought = wrap_thoughts("raw reasoning…", topic="graphs")
fixed = lint_markdown(doc.body, auto_fix=True)  # mordant, guarded
```

## Module layout

```
bobine/
├── __init__.py      public API
├── config.py        ConverterConfig, RoutingMode
├── engine.py        OnnxRapidEngine (lazy ONNX model manager)
├── converter.py     HybridConverter (core PDF/Office pipeline)
├── tables.py        HTML table → GFM pipe-table converter
├── assets.py        okf-asset:// staging for extracted images
├── versions.py      RapidAI version pins + runtime check
├── documents.py     Document model, frontmatter, thoughts wrapper
├── markdown.py      mordant linting (guarded, no-op without it)
└── pipeline.py      convert_to_markdown / stage_images / ingest_document
```

## Output contract

`ingest_document` produces a directory that a graph/import layer can consume:

- `<stem>.md` — linted markdown with `okf-asset://<id>` image links
- `_assets/<id>.<ext>` — staged image bytes (deduped, concept-scoped ids)

`bobine` never embeds, indexes, or writes to a database. The consumer owns
embedding and storage (in OKFgraph that is `OKFRouter.import_bundle`).

## Testing

```bash
# unit suite (no native backends needed — fake pdf_oxide objects drive the
# converter's routing/splice/ONNX-assembly paths)
pytest

# integration suite (requires bobine[pdf-ingest] installed)
pytest -m integration

# coverage + lint
pytest --cov=bobine --cov-report=term-missing
ruff check . && ruff format --check .
```

Markers: `integration` (real pdf_oxide/office_oxide/RapidAI) and `slow`
(heavy). The integration tests self-skip when backends are missing, so the
bare install always stays green. CI (`.github/workflows/ci.yml`) runs the
core suite on Python 3.10–3.13 plus an integration job.

## Version pinning

RapidAI packages move fast; `check_rapid_versions()` warns on first import if
an installed version drifts from the known-good list. Silence with
`BOBINE_INGEST_ALLOW_UNPINNED=1` (the legacy `OKFGRAPH_INGEST_ALLOW_UNPINNED`
is still honoured).
