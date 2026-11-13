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

# ONNX Runtime libraries (located on `import bobine`, never imported):
pip install "bobine[cpu]"   # onnxruntime == 1.31.1 (CPU)
pip install "bobine[gpu]"   # onnxruntime-gpu + NVIDIA CUDA/cuDNN wheels
```

GPU note: the `gpu` extra must include the `[cuda,cudnn]` wheels (it does —
a bare `onnxruntime-gpu` ships no `cudnn64_9.dll` and its CUDA EP dies at
the first Conv node, falling back silently). `import bobine` preloads the
NVIDIA DLLs into the process so the engine's direct library load resolves
them; verified live (pip ORT 1.30.0 + RTX 3090: scanned page OCR 28 → 883
chars). Never install both `onnxruntime` and `onnxruntime-gpu` (same module
name).

Requirements: Rust toolchain (1.85+, edition 2024), maturin ≥1.7, and a
modern onnxruntime (≥1.19) discoverable by `ort`. Python 3.10+
(prebuilt wheels for CPython 3.10–3.14 on manylinux x86_64/aarch64,
Windows x64, macOS arm64 — other platforms fall back to the sdist,
which needs the Rust toolchain above):

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
# binary files under text-ish extensions fail fast (UnsupportedFormat,
# content-sniffed) instead of returning decode garbage
```

## Public API (`import bobine`)

| Symbol | Purpose |
|---|---|
| `HybridConverter(config, cache_dir)` | Direct converter: `convert()`, `convert_pdf()`, `recognize_formula()`, `ingest_document()`, `describe()`, `close()` |
| `HybridConverter.ingest_document(path, out, on_page=, should_continue=)` | Full pipeline on a converter you hold: sessions stay warm across files (0.9) |
| `HybridConverter.close()` / `with` | Release the ONNX sessions; a closed converter refuses work until you build a new one |
| `HybridConverter.convert_office(path)` | Office → markdown (associated fn, no staging; `convert()` stages) |
| `convert_to_markdown(path, config, work_dir, cache_dir)` | Raw markdown string, one-shot |
| `convert_excel(path)` | Workbook → `ExcelDocument`: `.markdown`, `.sheet_names`, `.csv_by_sheet`, `.json` |
| `convert_jats(path, work_dir, config?)` | JATS `.xml`/`.jats` → `JatsDocument` (no models, no ORT; see below) |
| `HybridConverter.convert_jats(path, work_dir)` | Same, as a converter method |
| `ingest_document(path, output_dir, ...)` | Staged bundle → `ConvertedDocument` (`.md_text`, `.data_files`, lint/hooks; `cache_dir=None` = standard hub cache) |
| `model_status(cache_dir=None)` | Offline per-family cache status (TexTeller variant, layout, OCR, SLANet) — same dict keys as okfgraph's `model_info` |
| `runtime_info()` | What bobine decided about ONNX Runtime at import: `dylib`, `source` (`env` / `pip:<dist>` / `os`), `version`, `set_by`, `cuda_dirs`. `ORT_DYLIB_PATH` is the same decision, flattened |
| `convert_directory(source_dir, output_dir, ...)` | Batch ingest |
| `ConverterConfig` | Tuning knobs (see table below) |
| `RoutingMode` | `Never` / `Auto` / `Surgical` / `Always` |
| `FormulaBackend` | `TexTeller` (only backend) |
| `ModelPrecision` | `Fp32` / `Fp16` (`*_fp16.onnx`, shipped in the unified repo, 28-29/30 parity) |
| `ModelQuantization` | `Fp32` (default) / `Int8` (formula weights) |
| `DEVICES` / `PRECISIONS` | The accepted `device` / `precision` words (tuples) |
| `HybridConverter.describe()` | Per-pass plan: `providers`, `policy`, `precision`, `landed` |

## ONNX Runtime discovery

`import bobine` runs embroider's locator — the same one every embroider
model open and okfgraph use — before the first session, so the whole stack
loads one runtime instead of one per package. It reads the installed
`onnxruntime` dist's metadata and looks inside the package folder; it
never imports the module (importing it would load a second copy).

| Order | Wins when | `source` |
|---|---|---|
| 1 | `ORT_DYLIB_PATH` is already set | `env` |
| 2 | `onnxruntime` / `onnxruntime-gpu` is installed where Python would import it (`sys.path`, user site, editable installs) | `pip:<distribution>` |
| 3 | nothing found — `ort` would try the OS loader path, which can hold a stale copy | `os` |

`runtime_info()` reports the decision (`set_by` says who chose: `env`,
`bobine`, `none`), and the same import puts the CUDA lane's DLL folders on
`PATH` (`cuda_dirs`), which is what keeps a GPU build from failing on
`cudnn64_9.dll` and silently falling back to CPU. When nothing is found,
`dylib` is `None` and *nothing else happens* — no CUDA prep, no probe, no
loader call — so a machine whose only `onnxruntime` is a stale copy on the
OS loader path reports the fact quietly instead of panicking. Refuse to
open an ONNX pass there; the fast Office/text paths still work. The search and the CUDA
preparation are Rust (`src/runtime.rs` over `embroider::diag`); the Python
layer only passes `sys.path` and mirrors the choice into `os.environ`.


## Warm converter and the host (0.9)

One converter can ingest many documents. Sessions are built lazily on the
first document that needs them and stay open until you release them:

```python
conv = bobine.HybridConverter(bobine.ConverterConfig(), cache_dir="~/.cache/bobine")
for pdf in pdfs:
    doc = conv.ingest_document(pdf, f"out/{pdf.stem}")
conv.close()                    # or: with bobine.HybridConverter(...) as conv:
```

**The caller owns the lifetime.** Dropping the object frees the sessions
too; `close()` is the deterministic way to say so. It is idempotent, and
afterwards the converter refuses work rather than silently reloading
(construct a new one). One document at a time: a concurrent call raises
PyO3's borrow error instead of corrupting state, so callers serialise —
okfgraph holds a lock, the host serialises per settings key.

Out of process, `python -m bobine.serve` serves one converter per settings
key and keeps it warm across requests:

```
python -m bobine.serve --port 8757 --idle-seconds 300
```

| Route | Body -> answer |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /v1/models` | cache status + the warm keys (`settings`, `last_used`, `documents`, `describe()`) |
| `POST /v1/convert` | `{"path", "output_dir", "settings"?}` -> NDJSON: one `{"page", "total"}` line per page, then `{"md_path", "image_dir", "page_count", ...}` |

`settings` takes `ConverterConfig` kwargs (the request fields are
whitelisted, so a typo cannot poison the pool key). `--idle-seconds 0`
releases the converter after every conversion. A failed conversion
answers 200 with a final `{"error": …, "path": …}` line, the text a
consumer maps onto its own conversion error; bad requests and unusable
settings answer 400.


## RoutingMode

| Mode | Behaviour | Models loaded |
|---|---|---|
| `Never` | Fast path only (pdf_oxide). No ONNX models loaded. | none |
| `Auto` *(default)* | Per-page heuristics (math signal / scanned) → full layout+OCR only on flagged pages. | TexTeller + layout + OCR (table lazy on first scanned table) |
| `Surgical` | Fast path + formula crops via TexTeller; full pipeline only for scanned pages. | TexTeller only |
| `Always` | Every page through full layout + OCR, with hybrid formula refinement. | TexTeller + layout + OCR (table lazy) |

## Device and precision (the stack vocabulary)

`device` and `precision` are the same words embroider and okfgraph use, so
one row means one thing in every package. The accepted values are exposed
as `bobine.DEVICES` / `bobine.PRECISIONS` — read them instead of copying
the list.

```python
cfg = bobine.ConverterConfig(device="cuda", precision="fp16")
bobine.DEVICES     # ('auto', 'cpu', 'cuda')
bobine.PRECISIONS  # ('auto', 'fp32', 'fp16', 'int8')
```

| `device` | layout / OCR | table / POTATR | TexTeller |
|---|---|---|---|
| `auto` (default) | CUDA when the library registers it | CPU-pinned (measured 2-9x slower on CUDA) | base `ort_providers` |
| `cpu` | CPU | CPU | CPU |
| `cuda` | CUDA + CPU fallback | CUDA + CPU fallback (the pin is a measurement, not a limit) | CUDA + CPU fallback |

A per-pass provider list (`layout_ort_providers`, `table_ort_providers`, …)
always wins over the word — it is the expert override. `precision` maps
onto the older `model_precision` / `model_quantization` fields and wins
when set; setting both is refused at construction. `q8` is refused by
name: bobine has no q8 artifact, `int8` is the compact pair.

`HybridConverter.describe()` reports the decision per pass, before and
after the models load:

```python
conv.describe()
# {'device_requested': 'auto', 'precision_requested': 'auto',
#  'passes': {'encoder': {'providers': ['cpu'], 'policy': 'auto',
#                         'precision': 'fp32', 'landed': False},
#             'layout':  {'providers': None, 'policy': 'auto-cuda',
#                         'precision': 'fp32', 'landed': False},
#             'table':   {'providers': ['CPUExecutionProvider'],
#                         'policy': 'cpu-pinned', 'precision': 'fp32',
#                         'landed': False}, ...}}
```

`providers = None` means the answer depends on the loaded runtime (the
auto-CUDA slots) and nothing has opened yet — probing would load ONNX
Runtime just to describe a future load. Once a pass lands, its list is the
one the session opened with, and `landed` becomes `True`.


## Office documents

`.docx` / `.xlsx` / `.pptx` + legacy `.doc` / `.xls` / `.ppt` → markdown via
`office_oxide` (`convert_office`, or `convert()` by extension). No models
needed. Embedded pictures stage into `<work_dir>/assets/office/` with content-hash
names and links rewrite to the staged files (unreferenced ones gallery-appended,
like PDF figures); `ingest_document` promotes them to `okf-asset://`.
`extract_images: false` leaves the upstream markdown untouched (no staging,
no link rewriting — since office_oxide 0.1.11 the untouched output carries
caption text without resolvable image links rather than raw `(rId…)` refs).

Excel workbooks (`.xls` / `.xlsx`) additionally export structured data:

```python
doc = bobine.convert_excel("book.xlsx")
doc.markdown          # ## {sheet} sections + GFM tables
doc.sheet_names       # ['Data', 'Summary']
doc.csv_by_sheet      # {'Data': 'Item,Count,...\n', ...} (empty sheets skipped)
import json; data = json.loads(doc.json)  # typed values, formulas kept
```

`ingest_document` on a workbook also writes `<stem>.<sheet>.csv` siblings +
`<stem>.json` next to the `.md` (see `ConvertedDocument.data_files`).

## ConverterConfig fields

| Field | Default | Meaning |
|---|---|---|
| `extract_images` | `True` | Save embedded page images to `work_dir` |
| `append_unreferenced_images` | `True` | Gallery-append images not referenced in text |
| `use_onnx` | `True` | Master switch for ONNX passes |
| `routing_mode` | `Auto` | See table above |
| `formula_backend` | `TexTeller` | Formula recognizer |
| `table_backend` | `Slanet` | `Slanet` (default) or `Potatr` (opt-in page-level table recognition; weights auto-download, CPU-pinned like SLANet) |
| `potatr_node_threshold` | `0.8` | Min node score for a POTATR table claim |
| `potatr_edge_threshold` | `0.5` | Edge-link threshold for structure refinement |
| `potatr_suppress_iou` | `0.5` | IoU at which a POTATR claim suppresses an overlapping layout region |
| `potatr_min_text_fill` | `0.2` | Min fraction of claimed cells carrying text (textless claims fall back to region arms) |
| `device` | `"auto"` | Where every pass runs: `auto` (each slot's measured policy), `cpu`, or `cuda`. `cuda` overrides the table slots' measured CPU pin |
| `precision` | `"auto"` | Weight precision: `auto`, `fp32`, `fp16`, `int8`. Maps onto `model_precision`/`model_quantization` and **is authoritative when set**; `q8` is refused by name (no such artifact) |
| `model_precision` | `Fp32` | Legacy name for the formula weights; selects `*_fp16.onnx` when `Fp16`. Use `precision` instead |
| `model_quantization` | `Fp32` | Legacy name for the formula weights: `Fp32` (default) = reference output on CPU and CUDA; `Int8` = ~4x smaller RAM and ~1.9x faster on CPU, with occasional decode drift. Do not pair `Int8` with CUDA. Use `precision` instead |
| `render_dpi` | `300` | Renders for scanned-page OCR / layout |
| `formula_dpi` | `200` | Renders for formula crops |
| `detect_headings` | `True` | `#` headings from fast path |
| `structured_tables` | `True` | Grid-extract table regions before text-dump fallback |
| `min_figure_area_pts` | `100.0` | Smaller embedded images count as decoration |
| `image_output_dir` | `"assets"` | Asset tree root: `<dir>/p{page}/img{k}.{ext}` |
| `image_mode` | `Link` | `Link` (default) or `EmbedBase64` (data URIs ≤ cap) |
| `max_embed_bytes` | `262144` | Cap for `EmbedBase64`; larger figures stay links |
| `preserve_layout` | `False` | Fast path: strict visual order, no reflow (ONNX paths ignore it) |
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
| `sanitize_ocr` | `True` | Post-OCR LaTeX repair (fence fixup, `\left`/`\right` balance, script split); set `False` to get raw OCR text |
| `render_gate_enabled` | `True` | Render-health gate master switch (Option C; active only while built against pdf_oxide < 0.3.79, upstream #3098) |
| `render_gate_min_layer_words` | `30` | Min text-layer words for the gate to judge a page (below = scanned/thin, gate passes) |
| `render_gate_min_jaccard` | `0.35` | Fire when OCR↔layer word-Jaccard is strictly below this (page falls back to fast path, math suppressed) |
| `render_gate_salvage_inline_math` | `False` | Salvage inline-only formula boxes from the text layer as `$…$` on fired pages instead of dropping all math |

## Structured sources: JATS (v0.7.0)

Publisher full-text XML (`.xml` / `.jats` — PLOS, PMC, EuropePMC) bypasses
rendering/OCR entirely: MathML → LaTeX (`mml.rs`), section headings, GFM
tables, figures as remote-`href` links. Needs no models and no ORT.
`convert()` routes `.xml`/`.jats` here automatically.

```python
doc = bobine.convert_jats("paper.xml", work_dir="/tmp/out")
doc.md_path     # <work_dir>/<stem>.md (written by the call)
doc.title       # first `#` line, marker stripped
doc.n_formulas  # MathML-backed formula elements (health signal, not a display count:
                # inline `$…$` from non-MathML markup is not counted)
doc.n_figures   # <fig> floats
doc.n_tables    # <table-wrap> grids
```

Errors are `RuntimeError` with stable prefixes: `JATS error: XML parse: …`
(malformed XML), `JATS error: not a JATS article` (root isn't `<article>`).
Output dollars are guaranteed balanced (conversion fails otherwise).

## OCR sanitizer (v0.6.1)

ONNX-path markdown passes through `sanitize.rs` when
`text.sanitize_ocr` is set (default on): unclosed-fence fixup,
`\\left`/`\\right` balancing, consecutive-script splitting
(`x^a^b` → `x^a{}^b`). Disable only to debug raw OCR output.

## Common tasks

```python
# scanned document — force full pipeline at high dpi
bobine.ConverterConfig(routing_mode=bobine.RoutingMode.Always, render_dpi=300)

# fastest: text layer only, never touch ONNX
bobine.ConverterConfig(routing_mode=bobine.RoutingMode.Never, use_onnx=False)

# formulas on text-layer-hostile PDFs (Word/InDesign/OCR output)
bobine.ConverterConfig(routing_mode=bobine.RoutingMode.Surgical,
                       formula_layout_fallback=True)

# fp16 formula weights: the CUDA alternative to fp32 (half the size,
# ~1.2x faster on long decodes, 29/30 exact parity). ~3x slower on CPU.
bobine.ConverterConfig(model_precision=bobine.ModelPrecision.Fp16)

# int8 formula weights: the CPU alternative (~303 MB instead of ~1.19 GB,
# ~1.9x faster, with occasional decode drift):
bobine.ConverterConfig(model_quantization=bobine.ModelQuantization.Int8)

# GPU: layout and OCR auto-enable CUDA when the loaded library exposes a
# usable CUDA EP (EP-availability probe, measured 12.3x / 3.6x on an RTX
# 3090), while table recognition stays CPU-pinned (SLANet measures 2-9x
# slower on CUDA - its graph fragments across devices). Same build, no
# rebuild; degrades gracefully to CPU otherwise.
bobine.ConverterConfig()                # zero-config: the measured policy
bobine.ConverterConfig(device="cuda")   # every pass on CUDA, incl. tables
bobine.ConverterConfig(device="cpu")    # every pass on CPU, no EP probing

# Formula decode on GPU: fp32 is the default; route TexTeller through CUDA
# (~2.4x faster on the short fixture, ~1.35x on long decodes - RTX 3090).
# Int8 is CPU-oriented (quantized ops run slower on the GPU EP) - do NOT
# combine Int8 with CUDA; bobine logs a warning when it sees that.
bobine.ConverterConfig(
    precision="fp32", device="cuda")     # 0.9 spelling
bobine.ConverterConfig(
    ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# CUDA speed alternative: fp16 (half the weights, ~1.2x faster on long
# decodes, 29/30 exact parity with fp32).
bobine.ConverterConfig(
    model_precision=bobine.ModelPrecision.Fp16,
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

# macOS (Apple Silicon): opt into the Neural Engine via CoreML.
# Needs a CoreML-enabled onnxruntime dylib in ORT_DYLIB_PATH;
# falls back to CPU otherwise (CUDA entries stay dormant on mac).
bobine.ConverterConfig(ort_providers=["coreml", "CPUExecutionProvider"])

# ingest with lint + progress + cancellation callbacks
bobine.ingest_document("paper.pdf", "out/",
                       lint_callback=lambda md: (True, md.strip() + "\n"),
                       on_page=lambda i, n: print(f"page {i+1}/{n}"),
                       should_continue=lambda: not cancelled())
```

## Formula accuracy (measured)

30 display formulas from the ikat ground-truth set, rendered at 300 dpi,
greedy decode, RTX 3090 / Ryzen 9 5950X. Parity is exact match against fp32
after stripping display fences, `\left`/`\right`, `\limits`, and whitespace:

| Config | Exact GT | Similarity | Parity vs fp32 | Median |
|---|---|---|---|---|
| Fp32 + CPU (default) | 1/30 | 0.579 | — | 2.01 s |
| Int8 + CPU | 2/30 | 0.553 | 10/30 (0.83) | 1.05 s |
| Fp32 + CUDA | 1/30 | 0.579 | — | 1.49 s |
| Fp16 + CUDA | 1/30 | 0.579 | 29/30 (1.00) | 1.21 s |

Exact GT is low because TexTeller's house style differs from the
MathML-derived ground truth (`\mathrm`, `\mbox`, `\,` placement); parity is
the meaningful quantization metric. Int8 is ~1.9x faster on CPU but its
output drifts (mostly stylistic, occasionally a wrong formula); fp16 differs
from fp32 only in thin-space placement. Full matrix in
[benchmarks.md](benchmarks.md).

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
| TexTeller encoder + decoder + tokenizer (fp32, default) | auto-download from HuggingFace `opticsWolf/texteller3-onnx` into the hub cache | ~1.19 GB |
| TexTeller encoder + decoder fp16 | same unified repo (`*_fp16.onnx` pair, pure-fp16 compute; measured 28-29/30 exact parity with fp32) | ~598 MB |
| RapidLayout (PP-DocLayout_plus-L) | auto-download from HF `opticsWolf/pp-doclayout-plus-l-onnx` (Apache-2.0) into the hub cache, or `set_layout_model` | ~130 MB |
| RapidOCR PP-OCRv5 det + rec + dict | auto-download from HF `opticsWolf/rapidocr-ppocrv5-onnx` (Apache-2.0) into the hub cache, or `set_ocr_models` | ~22 MB |
| RapidTable (SLANet-plus) | auto-download from HF `opticsWolf/slanet-plus-onnx` (Apache-2.0) into the hub cache, or `set_table_model` | ~7.8 MB |
| POTATR page tables (opt-in via `table_backend`) | auto-download from HF `opticsWolf/potatr-onnx` (MIT) into the hub cache, or `set_potatr_model` | ~116 MB |
| TexTeller Int8 (opt-in CPU speed tier) | auto-download from HF `opticsWolf/texteller3-onnx` into the hub cache (KV-cache-capable merged graph) | ~303 MB |

`cache_dir` is the hub-cache root (an override of the standard HF
cache, shared machine-wide); `model_status(cache_dir=None)` reports
per-family offline status without loading anything.

Missing layout/OCR models degrade to the fast path per page; a missing
table model only disables scanned-table recognition.

Ownership: bobine fetches (own HF clients, own repo constants) and runs
(own `ort` sessions) every model above. embroider contributes the session
policy (`SessionPolicy::ort_defaults()`), provider fallback, the CUDA
probe, and — since 0.9 — the one runtime locator and CUDA-DLL preparation
(`locate_runtime` + `prepare_cuda_dirs`); it never sees converter weights.

## Testing

```bash
cargo test                 # 192 lib (ORT_DYLIB_PATH required) + office/excel/golden suites
ORT_DYLIB_PATH=... cargo test --test test_converter   # incl. full-paper AUTO run
maturin develop && python -m unittest discover -s python/tests   # 50 bindings tests (stdlib, no pytest)
```

The bindings tests cover what only an interpreter can show: a fresh
process imports bobine and `onnxruntime` stays out of `sys.modules` while
the library is still found; an explicit `ORT_DYLIB_PATH` wins and reports
`set_by="env"`; a missing runtime is data, not an error. Then the 0.9
surfaces: the device/precision words and their resolution, `describe()`
before and after sessions exist, the warm converter's reuse/`close()`/
`with`, the host's pool rules and every route (against a stub, plus one
real model-free conversion), the stub matching the real object, and the
site badge tracking `Cargo.toml`. The
decisions themselves live in Rust (`src/runtime.rs`, `config.rs`), with
unit tests there; Python only asks for them.

Fixtures: `tests/fixtures/` — CC BY 4.0 arXiv papers (attribution in
`SOURCES.md`) + generated scanned page + generated OOXML fixtures
(`examples/gen_office_fixtures.rs`).

## Benchmarks

Measured CPU/CUDA numbers (RTX 3090, ORT 1.28.1), TexTeller
fp32/fp16/q8/int8 timings and parity, and the `ORT_DYLIB_PATH` /
stale-System32-dll pitfall are documented in
[benchmarks.md](benchmarks.md).


## Env vars

| Var | Effect |
|---|---|
| `ORT_DYLIB_PATH` | Explicit onnxruntime shared library for `ort` (≥1.28 required, else `BadVersion`). **Usually unset**: `import bobine` auto-locates pip `onnxruntime`/`onnxruntime-gpu` (`pip install bobine[cpu]` / `bobine[gpu]`); set only to override (custom CUDA build). `bobine.ORT_DYLIB_PATH` shows the resolved path (None = not installed). |

## License

Apache-2.0 **OR** MIT (dual, choose either). Corpus fixtures are CC BY 4.0
(attribution in `tests/fixtures/SOURCES.md`). `legacy/` keeps the original
Python release's licensing.
