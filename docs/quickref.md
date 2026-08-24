# bobine — Quick Reference

Standalone PDF / Office / text → Markdown ingestion engine. Rust core with
Python bindings; the Python package imports with **zero ML dependencies**
(no torch, no optimum, no opencv) and degrades gracefully when optional
models are missing.

> Template: `legacy/docs/quickref.md` (pure-Python bobine v0.2.0).

## Install

```bash
# from the repo root (Rust workspace + maturin)
maturin develop --release          # editable install into current venv
# or build a wheel:
maturin build --release && pip install target/wheels/bobine-*.whl
```

Requirements: Rust toolchain (1.85+, edition 2024), maturin ≥1.7, and a
modern onnxruntime (≥1.19) discoverable by `ort`:

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime.dll   # e.g.
# <venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll
```

## One-liners

```python
import bobine

# single PDF → markdown string
config = bobine.ConverterConfig(routing_mode=bobine.RoutingMode.Surgical)
conv = bobine.HybridConverter(config, cache_dir="~/.cache/bobine")
md = conv.convert_pdf("paper.pdf", work_dir="/tmp/out")

# one-shot helper
md = bobine.convert_to_markdown("paper.pdf", config, work_dir="/tmp/out",
                                cache_dir="~/.cache/bobine")

# formula crop → LaTeX
latex = conv.recognize_formula("crop.png")     # r"\frac{1}{2}"

# text files need no models at all
md = conv.convert("notes.md", work_dir="/tmp/out")
```

## Public API (`import bobine`)

| Symbol | Purpose |
|---|---|
| `HybridConverter(config, cache_dir)` | Direct converter: `convert()`, `convert_pdf()`, `recognize_formula()` |
| `convert_to_markdown(path, config, work_dir, cache_dir)` | Raw markdown string, one-shot |
| `ConverterConfig` | Tuning knobs (see table below) |
| `RoutingMode` | `Never` / `Auto` / `Surgical` / `Always` |
| `FormulaBackend` | `TexTeller` (only backend) |
| `ModelPrecision` | `Fp32` / `Fp16` (`*_fp16.onnx`, models deferred) |

## RoutingMode

| Mode | Behaviour |
|---|---|
| `Never` | Fast path only (pdf_oxide). No ONNX models loaded. |
| `Auto` *(default)* | Per-page heuristics (math signal / scanned) → full layout+OCR only on flagged pages. |
| `Surgical` | Fast path + formula crops via TexTeller; full pipeline only for scanned pages. |
| `Always` | Every page through full layout + OCR. |

## ConverterConfig fields

| Field | Default | Meaning |
|---|---|---|
| `extract_images` | `True` | Save embedded page images to `work_dir` |
| `append_unreferenced_images` | `True` | Gallery-append images not referenced in text |
| `use_onnx` | `True` | Master switch for ONNX passes |
| `routing_mode` | `Auto` | See table above |
| `formula_backend` | `TexTeller` | Formula recognizer |
| `model_precision` | `Fp32` | Selects `*_fp16.onnx` when `Fp16` |
| `model_quantization` | `Int8` | Formula weights: `Int8` (default) = 4x smaller RAM, same CPU speed; `Fp32` = exact + KV-cache decode |
| `render_dpi` | `300` | Renders for scanned-page OCR / layout |
| `formula_dpi` | `200` | Renders for formula crops |
| `detect_headings` | `True` | `#` headings from fast path |
| `convert_html_tables` | `True` | HTML tables → GFM pipes |
| `min_formula_math_chars` | `5` | Min math chars for a text-layer formula box |
| `formula_inline_max_width_pts` | `220.0` | Box wider → `$$…$$` display math |
| `formula_pad_pts` | `4.0` | Padding around formula crops |
| `formula_layout_fallback` | `False` | Layout `equation` regions when text layer has no math fonts |
| `math_char_threshold` | `30` | Auto: math chars before flagging a page |
| `scanned_text_threshold` | `50` | Auto: max chars for scanned-page detection |
| `ocr_lang` | `"en"` | CTC charset fallback when the rec model lacks metadata |

## Common tasks

```python
# scanned document — force full pipeline at high dpi
bobine.ConverterConfig(routing_mode=bobine.RoutingMode.Always, render_dpi=300)

# fastest: text layer only, never touch ONNX
bobine.ConverterConfig(routing_mode=bobine.RoutingMode.Never, use_onnx=False)

# formulas on text-layer-hostile PDFs (Word/InDesign/OCR output)
bobine.ConverterConfig(routing_mode=bobine.RoutingMode.Surgical,
                       formula_layout_fallback=True)

# FP16 model variants (when present next to the fp32 files)
bobine.ConverterConfig(model_precision=bobine.ModelPrecision.Fp16)

# compact-memory formula recognition (~316 MB instead of ~1.25 GB):
bobine.ConverterConfig(model_quantization=bobine.ModelQuantization.Int8)

# ingest with lint + progress + cancellation callbacks
bobine.ingest_document("paper.pdf", "out/",
                       lint_callback=lambda md: (True, md.strip() + "
"),
                       on_page=lambda i, n: print(f"page {i+1}/{n}"),
                       should_continue=lambda: not cancelled())
```

## Models

| Model | Source | Size |
|---|---|---|
| TexTeller encoder + decoder + tokenizer | auto-download from HuggingFace `OleehyO/TexTeller` into `cache_dir` | ~1.25 GB |
| RapidLayout (DocLayout-YOLO) | local path via `OnnxEngine::set_layout_model` | ~30 MB |
| RapidOCR det + rec | local paths via `OnnxEngine::set_ocr_models` | ~15 MB |
| RapidTable (SLANet-plus) | auto-download from HF `opendatalab/PDF-Extract-Kit-1.0` into `<cache>/models/`, or `set_table_model` | ~7.8 MB |

Missing layout/OCR models degrade to the fast path per page; a missing
table model only disables scanned-table recognition.

## Testing

```bash
cargo test                 # 45 tests (unit + integration)
ORT_DYLIB_PATH=... cargo test   # needed for the PDF integration tests
maturin develop && python -c "import bobine"   # bindings smoke test
```

Fixtures: `tests/fixtures/` — CC BY 4.0 arXiv papers (attribution in
`SOURCES.md`) + generated scanned page.

## Env vars

| Var | Effect |
|---|---|
| `ORT_DYLIB_PATH` | Explicit onnxruntime shared library for `ort` (≥1.19 required) |

## License

Apache-2.0 **OR** MIT (dual, choose either). Corpus fixtures are CC BY 4.0
(attribution in `tests/fixtures/SOURCES.md`). `legacy/` keeps the original
Python release's licensing.
