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

| Mode | Behaviour | Models loaded |
|---|---|---|
| `Never` | Fast path only (pdf_oxide). No ONNX models loaded. | none |
| `Auto` *(default)* | Per-page heuristics (math signal / scanned) → full layout+OCR only on flagged pages. | TexTeller + layout + OCR (table lazy on first scanned table) |
| `Surgical` | Fast path + formula crops via TexTeller; full pipeline only for scanned pages. | TexTeller only |
| `Always` | Every page through full layout + OCR, with hybrid formula refinement. | TexTeller + layout + OCR (table lazy) |

## ConverterConfig fields

| Field | Default | Meaning |
|---|---|---|
| `extract_images` | `True` | Save embedded page images to `work_dir` |
| `append_unreferenced_images` | `True` | Gallery-append images not referenced in text |
| `use_onnx` | `True` | Master switch for ONNX passes |
| `routing_mode` | `Auto` | See table above |
| `formula_backend` | `TexTeller` | Formula recognizer |
| `model_precision` | `Fp32` | Selects `*_fp16.onnx` when `Fp16` |
| `model_quantization` | `Int8` | Formula weights: `Int8` (default, Ji-Ha export) = 4x smaller RAM **and** ~2.4x faster than Fp32 on CPU (KV-cached); `Fp32` = reference output |
| `render_dpi` | `300` | Renders for scanned-page OCR / layout |
| `formula_dpi` | `200` | Renders for formula crops |
| `detect_headings` | `True` | `#` headings from fast path |
| `structured_tables` | `True` | Grid-extract table regions before text-dump fallback |
| `min_figure_area_pts` | `100.0` | Smaller embedded images count as decoration |
| `image_output_dir` | `"assets"` | Asset tree root: `<dir>/p{page}/img{k}.{ext}` |
| `promote_headings` | `True` | Scholarly section patterns (`I.`, `3.1`, `A.`) → markdown headings |
| `promote_title` | `True` | Document title → `# ` (first short block or top title-like heading) |
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

# GPU (dynamic): point ORT_DYLIB_PATH at a GPU-enabled ONNX Runtime library
# (onnxruntime-gpu >= 1.19 + matching CUDA/cuDNN). v0.4.9+ auto-enables CUDA
# for the layout and OCR slots when the loaded library registers the CUDA EP
# (measured 12.3x / 3.6x on an RTX 3090) and pins table recognition to CPU
# (SLANet measures 2-9x slower on CUDA - its graph fragments across devices).
# Same build, no rebuild needed; degrades gracefully to CPU otherwise.
bobine.ConverterConfig()   # zero-config GPU defaults

# Fp32 formula decode on GPU: also route TexTeller through CUDA
# (~2.3x faster formulas measured on an RTX 3090). Int8 is CPU-oriented
# (quantized ops run slower on the GPU EP) - do NOT combine Int8 with CUDA;
# bobine logs a warning when it sees that combination.
bobine.ConverterConfig(
    model_quantization=bobine.ModelQuantization.Fp32,
    ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# per-slot overrides (pin any slot, mix freely):
bobine.ConverterConfig(
    layout_ort_providers=["CPUExecutionProvider"],   # keep layout on CPU
    ocr_ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    table_ort_providers=["CUDAExecutionProvider"],   # measured: avoid
    encoder_ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    decoder_ort_providers=["CPUExecutionProvider"])

# or offload only the ViT encoder to the GPU and keep the autoregressive
# decoder on CPU (~20% faster than all-CPU):
bobine.ConverterConfig(
    encoder_ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# ingest with lint + progress + cancellation callbacks
bobine.ingest_document("paper.pdf", "out/",
                       lint_callback=lambda md: (True, md.strip() + "\n"),
                       on_page=lambda i, n: print(f"page {i+1}/{n}"),
                       should_continue=lambda: not cancelled())
```

## Formula accuracy (measured)

10-example corpus (matplotlib-rendered: Gaussian integral, Basel sum,
heat equation, Euler's identity, vector norm, binomial, AM-GM product,
limit, contour integral), greedy decode, RTX 3090 / Ryzen 9 5950X:

| Config | Correct | Miss | Median time |
|---|---|---|---|
| Int8 + CPU (default) | 10/10 | — | ~0.5 s |
| Fp32 + CUDA | 9/10 | dropped one token (`n` → `1` in a sum limit) | ~0.4 s |

The table reflects v0.4.4, after fixing the pad fill to background-white
in normalized space (upstream normalizes *before* padding; raw-black
padding caused e.g. `\oint` misreads on both variants).
Before the fix each variant missed a different example; afterwards the
misses flipped again. Conclusion: misses are single-token greedy-decode
noise near decision boundaries, not systematic quantization or precision
damage. **Int8 costs no measurable accuracy** - choose by hardware.
Raw fixtures in `%TEMP%/bobine_test/bench10/`.

## Formula routing in ALWAYS mode (v0.4.8)

The layout model localizes formulas only at paragraph granularity on dense
math pages (its `isolate_formula` boxes are half-page blobs at our 1024-side
letterbox - measured, not an artifact: identical at native resolution). v0.4.7
therefore degraded every born-digital formula region to plain text. v0.4.8
refines instead: each layout formula region is intersected with text-layer
math boxes, the surviving display-style crops get a budgeted TexTeller decode
(area-proportional token cap + plausibility filter), and the LaTeX is spliced
back into the region's text. On the 4-page fixture this makes ALWAYS output
byte-identical to SURGICAL while keeping scans (no text layer) on the full
crop path. Expected: ALWAYS ≈ SURGICAL cost + layout pass; no path may spend
more than ~2 s per formula crop.

## Models

| Model | Source | Size |
|---|---|---|
| TexTeller encoder + decoder + tokenizer | auto-download from HuggingFace `OleehyO/TexTeller` into `cache_dir` | ~1.25 GB |
| RapidLayout (DocLayout-YOLO) | auto-download from HF `wybxc/DocLayout-YOLO-DocStructBench-onnx` into cache dir, or `set_layout_model` | ~72 MB |
| RapidOCR PP-OCRv4 det + rec | auto-download from HF `SWHL/RapidOCR` into `<cache>/PP-OCRv4/`, or `set_ocr_models` | ~16 MB |
| RapidTable (SLANet-plus) | auto-download from HF `opendatalab/PDF-Extract-Kit-1.0` into `<cache>/models/`, or `set_table_model` | ~7.8 MB |
| TexTeller Int8 (default formula weights) | auto-download from HF `Ji-Ha/TexTeller3-ONNX-dynamic` into `<cache>/texteller_int8/` (KV-cache-capable merged graph) | ~319 MB |

Missing layout/OCR models degrade to the fast path per page; a missing
table model only disables scanned-table recognition.

## Testing

```bash
cargo test                 # 83 unit + 9 integration tests
ORT_DYLIB_PATH=... cargo test   # needed for the PDF integration tests
maturin develop && python -c "import bobine"   # bindings smoke test
```

Fixtures: `tests/fixtures/` — CC BY 4.0 arXiv papers (attribution in
`SOURCES.md`) + generated scanned page.

## Benchmarks

Measured CPU/CUDA numbers (RTX 3090, ORT 1.28.1), TexTeller fp32 vs int8
timings, and the `ORT_DYLIB_PATH` / stale-System32-dll pitfall are documented in
[benchmarks.md](benchmarks.md).


## Env vars

| Var | Effect |
|---|---|
| `ORT_DYLIB_PATH` | Explicit onnxruntime shared library for `ort` (≥1.19 required). Point at a **GPU build** (onnxruntime-gpu + CUDA/cuDNN) to activate `ort_providers=["cuda", ...]`; CPU builds degrade gracefully |

## License

Apache-2.0 **OR** MIT (dual, choose either). Corpus fixtures are CC BY 4.0
(attribution in `tests/fixtures/SOURCES.md`). `legacy/` keeps the original
Python release's licensing.
