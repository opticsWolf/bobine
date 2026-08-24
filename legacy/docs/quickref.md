# bobine — Quick Reference

Standalone PDF / Office / text → Markdown ingestion engine. Everything below
is the stable surface; the package imports with **zero dependencies** and
degrades gracefully when optional backends are missing.

## Install

```bash
pip install bobine                 # core (Pillow only)
pip install "bobine[pdf-ingest]"    # + pdf_oxide, office_oxide, RapidAI ONNX
pip install "bobine[formula]"       # + vendored formula OCR runtime
pip install "bobine[markdown]"      # + mordant linting, frontmatter parsing
pip install "bobine[dev]"           # + pytest, reportlab (corpus generator)
```

Extras compose freely. `[pdf-ingest]` + `[formula]` together = full pipeline.

## One-liners

```python
from bobine import ConverterConfig, RoutingMode, ingest_document, convert_directory

# single document → staged, linted markdown in ./out
r = ingest_document("paper.pdf", "out", config=ConverterConfig(routing_mode=RoutingMode.SURGICAL))
print(r.md_path, r.page_count, r.image_count, r.lint)

# batch a whole directory
results = convert_directory("docs/", "out/")

# text files need no native deps at all
ingest_document("notes.txt", "out")
```

## Public API (`from bobine import …`)

| Symbol | Purpose |
|---|---|
| `ingest_document(path, output_dir, *, config, lint, auto_fix, log)` | Full pipeline → `ConvertedDocument` (staged `_assets/`, linted md) |
| `convert_directory(source_dir, output_dir, *, config, log)` | Batch `ingest_document` over `rglob` |
| `convert_to_markdown(path, config, work_dir, *, should_continue, on_page, log)` | Raw markdown string only, no staging/lint |
| `stage_images(md, image_dir, *, output_dir)` | Rewrite local `![](...)` links to `okf-asset://` + copy bytes |
| `HybridConverter(config, log)` | Direct converter; `ensure_models()` / `convert_pdf()` / `convert_office()` / `close()` |
| `OnnxRapidEngine(log_fn, ort_providers)` | Lazy ONNX model manager (formula/ocr/layout/table) |
| `ConverterConfig`, `RoutingMode` | Tuning knobs + routing modes |
| `Document`, `load_markdown_document(path)`, `wrap_thoughts(text, topic)` | Text-doc model, frontmatter, thoughts wrapper |
| `lint_markdown(content, *, auto_fix)`, `lint_markdown_file(path)` | mordant linting (guarded, no-op `E999` without it) |
| `html_tables_to_gfm(html)` | HTML table → GFM pipe table |
| `check_rapid_versions()` | Version drift warning on import |
| `asset_id`, `stage_images_as_okf_assets`, `ASSET_STORE_DIRNAME` | Low-level staging primitives |

`ConvertedDocument`: `.md_path`, `.md_text`, `.image_dir`, `.image_count`,
`.page_count`, `.lint` (dict with `errors`/`fixed`/`content`).

## RoutingMode

| Mode | Behaviour |
|---|---|
| `NEVER` | Fast path only (pdf_oxide). No ONNX models loaded. |
| `AUTO` | Per-page heuristics (math signal / scanned) → full ONNX layout+OCR only on flagged pages. |
| `SURGICAL` | Fast path + formula crops (vendored RapidLaTeXOCR); full pipeline only for scanned pages. |
| `ALWAYS` | Every page through full ONNX layout + OCR. |

## ConverterConfig fields

| Field | Default | Meaning |
|---|---|---|
| `extract_images` | `True` | Stage embedded images as assets |
| `append_unreferenced_images` | `True` | Gallery-append images not referenced in text |
| `use_onnx` | `True` | Master switch for ONNX passes |
| `routing_mode` | `AUTO` | See table above |
| `ort_providers` | `None` | ONNX Runtime providers (`None` → from `device`) |
| `render_dpi` | `300` | Renders for scanned-page OCR / layout |
| `formula_dpi` | `200` | Renders for formula crops |
| `detect_headings` | `True` | `#` headings from title regions / fast path |
| `convert_html_tables` | `True` | slanet HTML → GFM pipes |
| `math_char_threshold` | `30` | AUTO: math chars before flagging a page |
| `scanned_text_threshold` | `50` | AUTO: max chars for scanned-page detection |
| `formula_batch_size` | `8` | Reserved (single-crop recognizer) |
| `formula_pad_pts` | `4.0` | Padding around formula crops |
| `min_formula_math_chars` | `5` | Min chars for a text-layer formula box |
| `formula_inline_max_width_pts` | `220.0` | Box wider → `$$…$$` display math |
| `formula_layout_fallback` | `False` | P2: layout `equation` regions when text layer has no math fonts |
| `detect_code_blocks` | `True` | Fence monospace runs |
| `rescue_bad_tables` | `False` | Fast-path table repair (off) |
| `ocr_lang` | `"en"` | RapidOCR language |
| `device` | `"cuda"` | `"cuda"`/`"gpu"` → CUDA providers, else CPU |

## Common tasks

```python
# scanned / born-digital control
ConverterConfig(routing_mode=RoutingMode.ALWAYS, render_dpi=300)  # scanned docs
ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False)  # fast text only

# formulas on text-layer-hostile PDFs (Word/InDesign/OCR output)
ConverterConfig(routing_mode=RoutingMode.SURGICAL, formula_layout_fallback=True)

# CPU-only
ConverterConfig(device="cpu")

# text pipeline without a PDF backend
doc = load_markdown_document("note.md")
fixed = lint_markdown(doc.body, auto_fix=True)
```

## Testing

```bash
pytest                          # unit suite, no native backends (fake pdf_oxide)
pytest -m integration           # real backends + PDF corpus (needs [pdf-ingest])
pytest -m slow                  # ONNX runs over real pages
pytest --cov=bobine             # 92 % coverage
ruff check . && ruff format --check .
```

Markers: `integration`, `slow`. Corpus fixtures in `tests/fixtures/pdf/`
(CC BY 4.0 arXiv trims + generated scanned page — see `SOURCES.md`).

## Env vars

| Var | Effect |
|---|---|
| `BOBINE_INGEST_ALLOW_UNPINNED=1` | Silence RapidAI version-drift warning |
| `OKFGRAPH_INGEST_ALLOW_UNPINNED=1` | Legacy alias, still honoured |

## License

Apache-2.0 **OR** MIT (dual, choose either). Vendored `bobine/_vendor/`
code keeps its own license (MIT (c) 2023 RapidAI). Corpus fixtures are CC
BY 4.0 (attribution in `tests/fixtures/SOURCES.md`).
