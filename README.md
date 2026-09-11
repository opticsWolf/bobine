# bobine

[![Crates.io](https://img.shields.io/crates/v/bobine)](https://crates.io/crates/bobine)
[![docs.rs](https://img.shields.io/docsrs/bobine)](https://docs.rs/bobine)
[![PyPI](https://img.shields.io/pypi/v/bobine)](https://pypi.org/project/bobine/)
[![Python](https://img.shields.io/pypi/pyversions/bobine)](https://pypi.org/project/bobine/)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)](https://www.rust-lang.org)
[![onnxruntime](https://img.shields.io/badge/onnxruntime-%E2%89%A51.19-blue)](https://onnxruntime.ai)
[![CI](https://github.com/opticsWolf/bobine/actions/workflows/ci.yml/badge.svg)](https://github.com/opticsWolf/bobine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-green)](https://github.com/opticsWolf/bobine/blob/main/LICENSE)

Standalone **PDF / Office / text → Markdown ingestion engine** — a pure-Rust
core with Python bindings. Runs on a single `onnxruntime` shared library with
no CUDA-version coupling: `pdf_oxide` for fast native PDF text extraction,
ONNX models (TexTeller formula OCR, DocLayout-YOLO layout analysis, PaddleOCR)
for the heavy passes. **No torch, no optimum, no opencv** — not even on the
Python side.

Dual-licensed under the terms of either the MIT License or the Apache License,
Version 2.0 — you may choose either (see [LICENSE](https://github.com/opticsWolf/bobine/blob/main/LICENSE)).

## Docs

- [**Quick reference**](https://github.com/opticsWolf/bobine/blob/main/docs/quickref.md) — install, API, config, common tasks
- [**Architecture**](https://github.com/opticsWolf/bobine/blob/main/docs/architecture.md) — modules, data flow, coordinate spaces, model acquisition
- [**Benchmarks & test results**](https://github.com/opticsWolf/bobine/blob/main/docs/benchmarks.md) — CPU vs CUDA timings, TexTeller fp32/int8, environment setup
- [**Proposal: figures/tables/layout**](https://github.com/opticsWolf/bobine/blob/main/docs/proposal_media_tables.md) — plan for reading-order image placement and structured table extraction
- [**Implementation plan**](https://github.com/opticsWolf/bobine/blob/main/IMPLEMENTATION_PLAN.md) — status, gap inventory, phased roadmap
- [**Office export plan**](https://github.com/opticsWolf/bobine/blob/main/IMPLEMENTATION_PLAN_office.md) — md for all formats, Excel csv/json, picture extraction

## Why bobine?

The ingestion pipeline was originally entangled with the knowledge-graph
project it served (`OKFgraph`). This crate moves conversion into its own
package so any consumer — a graph, a CLI, an MCP server, a batch tool — can
reuse it without importing a database stack. The v0.3.0 rewrite ports the
whole pipeline to Rust: same routing heuristics and output contract as the
proven Python implementation (preserved under [`legacy/`](https://github.com/opticsWolf/bobine/tree/main/legacy)), with
native speed and no Python ML dependencies.

## Installation

```bash
# Python package (wheels: linux + windows x86_64, macOS arm64)
pip install bobine

# Rust library (crates.io, no Python involved)
cargo add bobine

# ...or build from source (repo root)
pip install maturin && maturin develop --release
```

Runtime requirement: an ONNX Runtime library for the heavy passes — or
nothing at all (Office/text/fast-path PDF work needs no models):

```bash
pip install bobine[cpu]   # adds onnxruntime >= 1.28 (CPU)
# or: pip install bobine[gpu]   # onnxruntime-gpu (self-contained CUDA)
```

`import bobine` points `ORT_DYLIB_PATH` at the pip-installed library
automatically; an already-set `ORT_DYLIB_PATH` always wins (e.g. a custom
CUDA build):

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime.dll   # e.g. <venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll
```

`ort` loads the library dynamically (`load-dynamic`, no CUDA-version
coupling) and refuses runtimes older than 1.28 (`BadVersion`) — hence the
`>=1.28` pins. Never install both `onnxruntime` and `onnxruntime-gpu`
(same module name, they clobber each other).

## Quick start (Python)

```python
import bobine

conv = bobine.HybridConverter(
    bobine.ConverterConfig(routing_mode=bobine.RoutingMode.Surgical),
    cache_dir="~/.cache/bobine",       # TexTeller models auto-download here
)

md = conv.convert_pdf("paper.pdf", work_dir="/tmp/out")
latex = conv.recognize_formula("crop.png")          # r"\frac{1}{2}"

# one-shot helper
md = bobine.convert_to_markdown(
    "paper.pdf", bobine.ConverterConfig(), work_dir="/tmp/out",
    cache_dir="~/.cache/bobine",
)

# Text documents need no models at all
md = conv.convert("notes.txt", work_dir="/tmp/out")
```

### PDF conversion with ONNX heavy passes

`HybridConverter` routes pages through four modes:

| Mode      | Behaviour                                                                 | Models loaded |
|-----------|---------------------------------------------------------------------------|---------------|
| `Never`   | Fast path only (pdf_oxide). No ONNX models loaded.                        | none |
| `Auto` *(default)* | Heuristics per page → full ONNX layout + OCR on flagged pages.   | TexTeller + layout + OCR (table lazy) |
| `Surgical`| Formula crops via TexTeller only; full pipeline just for scans.           | TexTeller only |
| `Always`  | Every page through the full ONNX layout + OCR pipeline; on born-digital pages, formula regions are refined against text-layer math boxes (v0.4.8) and output matches `Surgical`. | TexTeller + layout + OCR (table lazy) |

RapidLayout and RapidOCR weights also auto-download from HuggingFace on
first use (DocStructBench YOLO ~72 MB, PP-OCRv4 ~16 MB); explicit local
paths can override them. Missing/broken heavy models degrade gracefully to
the fast path per page — a conversion never fails because of them.

## Quick start (Rust)

```rust
use bobine::{ConverterConfig, HybridConverter, RoutingMode};

let mut conv = HybridConverter::new(
    ConverterConfig { routing_mode: RoutingMode::Surgical, ..Default::default() },
    std::path::Path::new("~/.cache/bobine"),
);
let md = conv.convert_pdf(std::path::Path::new("paper.pdf"),
                          std::path::Path::new("/tmp/out"))?;
```

## Module layout

```
src/
├── lib.rs            crate root & re-exports
├── config.rs         ConverterConfig · RoutingMode · ModelPrecision
├── engine.rs         OnnxEngine (lazy model manager: TexTeller/Layout/OCR)
├── converter.rs      HybridConverter (core PDF/Office pipeline)
├── tex_teller.rs     TexTeller ONNX — ViT encoder → RoBERTa decoder
├── rapid_layout.rs   DocLayout-YOLO page layout analysis
├── rapid_ocr.rs      PaddleOCR det + rec (DBNet / CRNN, CTC decode)
├── rapid_table.rs    SLANet-plus table-structure recognition (scans)
├── pdf_source.rs     PdfSource trait — page text without a real PDF file
├── tables.rs         HTML table → GFM pipe-table converter
├── excel.rs          Excel workbooks → per-sheet csv/json/md
├── office_images.rs  Office picture staging (IR walk + package fallback)
├── assets.rs         okf-asset://\ staging store
├── documents.rs      ConvertedDocument + frontmatter
├── pipeline.rs       ingest_document / convert_directory / ProgressHooks
├── error.rs          BobineError
└── py_bindings.rs    PyO3 surface (behind the extension-module feature)
python/bobine/        Python shim + type stubs        (import bobine)
legacy/               frozen pure-Python bobine v0.2.0 (reference implementation)
```

### Formula OCR (SURGICAL mode)

Formulas are recognized by **TexTeller** (80M training pairs), decoded with
KV-cache over a ViT encoder → RoBERTa decoder ONNX graph — roughly 5× faster
per crop than the RapidLaTeXOCR backend used by the legacy Python package,
with markedly better accuracy.

By default bobine downloads the quantized export
(~319 MB total, HF `Ji-Ha/TexTeller3-ONNX-dynamic`) into the converter's
cache dir on first use; set `model_quantization=ModelQuantization::Fp32` to
use full-precision weights (~1.25 GB, HF `OleehyO/TexTeller`) instead.
Preprocessing matches upstream TexTeller exactly, including its
normalize-before-pad order (v0.4.4 fixed black pad fill, which measurably
degraded recognition). Measured on a 10-formula corpus: Int8+CPU scores
10/10, Fp32+CUDA 9/10 — occasional single-token decode noise flips between
examples on either variant, so pick by hardware, not quality.

On NVIDIA GPUs, point `ORT_DYLIB_PATH` at a GPU onnxruntime build —
v0.4.9+ auto-enables CUDA for the layout and OCR slots when the loaded
library registers the CUDA execution provider (measured 12.3x / 3.6x
speedups), and keeps table recognition pinned to CPU (SLANet measures
2-9x slower on CUDA: its graph fragments across devices). Zero config
needed; explicit per-slot overrides: `layout_ort_providers`,
`ocr_ort_providers`, `table_ort_providers`, `encoder_ort_providers`,
`decoder_ort_providers`. To also run Fp32 formula decode on the GPU, set
`ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`
(Int8 + CUDA is discouraged and warned against).

Formula regions come from the PDF text layer (TeX math fonts such as
`cmmi`/`cmsy`/`cmex`, plus unicode math codepoints), merged **line-aware** so
multi-line display equations become one crop while separate equations,
columns and prose stay apart. For text-layer-hostile PDFs (Word/InDesign/OCR
output without math fonts), set `formula_layout_fallback=True` to ask the
layout model for equation regions instead (off by default).

OCR recognition runs line crops in chunks of 32 on accelerators (6.3x faster
on CUDA, measured) while CPU-only sessions keep the exact single-line path —
provider-gated batching, byte-identical tensors on CPU (v0.4.28, see
[benchmarks](https://github.com/opticsWolf/bobine/blob/main/docs/benchmarks.md)).

## Office documents

`.docx` / `.xlsx` / `.pptx` plus legacy `.doc` / `.xls` / `.ppt` convert to
markdown via `office_oxide` (`HybridConverter::convert_office(path)` or the
`convert()` dispatcher — no models needed). Excel workbooks additionally export per-sheet
csv/json (`convert_excel`, `<stem>.<sheet>.csv` + `<stem>.json` siblings).
Embedded pictures stage into `<work_dir>/assets/office/` and rewrite to staged
files, promoted to `okf-asset://` by `ingest_document` — details in the
[**Office export plan**](https://github.com/opticsWolf/bobine/blob/main/IMPLEMENTATION_PLAN_office.md).

## Output contract

`convert_pdf` returns one markdown string: inline `$…$` / display `$$…$$`
LaTeX spliced in place, GFM pipe tables, fenced code blocks from monospaced
font runs, embedded images written to `work_dir`, plus an `okf-asset://` staging store
for unreferenced figures. Office docs additionally stage pictures into
`<work_dir>/assets/office/`; workbooks yield `<stem>.<sheet>.csv` +
`<stem>.json` siblings. For document-level workflows use
`ingest_document` / `convert_directory`, which return versioned
`ConvertedDocument`s with frontmatter and lint hooks.

## Testing

```bash
cargo test                # 128 lib tests (ORT_DYLIB_PATH required — no dylib = abort)
cargo test --test test_office --test test_excel --test test_golden   # no models needed
ORT_DYLIB_PATH=... cargo test --test test_converter                 # incl. full-paper AUTO run
maturin develop && python -c "import bobine"   # bindings smoke test
```

Fixtures: trimmed CC BY 4.0 arXiv papers + a generated scanned page +
generated OOXML fixtures — see `tests/fixtures/SOURCES.md` for provenance.

## ONNX Runtime versioning

`ort` requires onnxruntime ≥ 1.19 and fails loudly at session creation on
older system libraries (`BadVersion`). Set `ORT_DYLIB_PATH` explicitly in CI
or dev environments with multiple installations. GPU support = point the same
variable at a CUDA-enabled build; layout/OCR then auto-enable CUDA per
slot (v0.4.9+) and `ConverterConfig` exposes per-slot overrides
(`layout_ort_providers`, `ocr_ort_providers`, `table_ort_providers`,
`encoder_ort_providers`, `decoder_ort_providers`), with `ort_providers`
as the base list for TexTeller.

## License

Apache-2.0 **OR** MIT (dual, choose either). Test fixtures are CC BY 4.0
(attribution in `tests/fixtures/SOURCES.md`). `legacy/` keeps the original
Python release's licensing.
