# bobine — Architecture

A standalone **PDF / Office / text → Markdown ingestion engine**, implemented
in Rust (`bobine_rs` crate) with Python bindings via PyO3/maturin
(`import bobine`). This document explains how the pieces fit, the data
flows, and the non-obvious invariants.

> Template: `legacy/docs/architecture.md` (pure-Python bobine v0.2.0).

---

## 1. Goals & non-goals

**Goals**
- Convert PDF (fast native text + optional ONNX heavy passes), Office
  (docx/xlsx/pptx/doc/xls/ppt), and text documents (txt/md) to markdown.
- Formula recognition via **TexTeller** ONNX (ViT encoder → RoBERTa decoder).
- Page layout analysis via **DocLayout-YOLO**; text OCR via **PaddleOCR**
  det+rec ONNX.
- Expose everything to Python with **zero Python ML dependencies** — no
  torch, no optimum, no opencv. The heavy lifting is pure Rust.

**Non-goals** (by design)
- No database, no embedding, no UI. The consumer owns storage/embedding.
- No RapidLaTeXOCR: TexTeller replaced it (5× faster per crop, better
  accuracy; costs ~1 GB more RAM).
- No CUDA-version coupling: `ort` uses `load-dynamic`; point
  `ORT_DYLIB_PATH` at any modern onnxruntime (≥1.19), CUDA included.

---

## 2. Module map

```
src/
├── lib.rs            crate root, re-exports
├── config.rs         ConverterConfig, RoutingMode, FormulaBackend, ModelPrecision
├── error.rs          BobineError enum
├── engine.rs         OnnxEngine — lazy model manager (TexTeller/RapidLayout/RapidOCR)
├── converter.rs      HybridConverter — core PDF/Office pipeline (~700 LOC)
│                     (internals generic over PdfSource)
├── pdf_source.rs     PdfSource trait: converter ↔ PDF backend seam;
│                     pdf_oxide adapter + in-memory FakePdf test fakes
├── tex_teller.rs     TexTeller ONNX: preprocess → encoder → autoregressive decode
├── rapid_layout.rs   DocLayout-YOLO: LetterBox(1024) → NMS → LayoutRegion
├── rapid_ocr.rs      PaddleOCR DBNet + CRNN: DB unclip (min-area rect +
│                     polygon offset), rotation-aware crops, CTC decode
├── rapid_table.rs    RapidTable (SLANet-plus): scanned-table → HTML
├── tables.rs         HTML <table> → GFM pipe-table converter
└── py_bindings.rs    PyO3 surface: bobine._native
python/bobine/        Python shim (__init__.py re-exports _native) + .pyi stubs
legacy/               frozen pure-Python bobine v0.2.0 (reference implementation)
```

Dependency direction: `py_bindings → converter → engine → (tex_teller |
rapid_layout | rapid_ocr)`; `converter → tables`, `converter → pdf_source`. No cycles.

---

## 3. Data flows

### `HybridConverter.convert_pdf(path, work_dir)`

```
open pdf_oxide::api::Pdf
  for each page i:
    extract embedded images (if extract_images)
    page_md = route_page(pdf, i, work_dir)        ← routing decision
    maybe append image gallery
  join pages with "---" separators
```

### Per-page routing (`route_page`)

```
NEVER    → fast path (pdf_oxide to_markdown)                    [no models]
SURGICAL → scanned?  → full_structure_page_markdown
             else   → surgical_page_markdown (formula crops)
AUTO     → needs_onnx(page)? yes → full structure; else fast path
ALWAYS   → full structure on every page
```

Routing signals (mirroring the Python heuristics):
- **Math signal**: text-layer chars whose font matches TeX math keywords
  (`cmmi`, `cmsy`, `cmex`, …) or unicode math ranges (Greek, U+2200–22FF,
  U+1D400–1D7FF, …) above `math_char_threshold`.
- **Scanned**: fewer than `scanned_text_threshold` chars + page has images.

### `convert()` dispatch

| Extension | Path |
|---|---|
| `.pdf` | `convert_pdf` |
| `.docx .xlsx .pptx .doc .xls .ppt` | `office_oxide::Document::open().to_markdown()` (auto-detect) |
| anything else | raw UTF-8 read |

---

## 4. OnnxEngine — lazy loading & degradation

Three lazy slots, each loaded on first use and cached:

| Slot | Model source | Loaded by |
|---|---|---|
| `tex_teller` | HuggingFace `OleehyO/TexTeller` via `hf-hub` (blocking) | `ensure_tex_teller()` |
| `layout` | local ONNX path (`set_layout_model()`) | `ensure_layout()` |
| `ocr` | local det/rec ONNX paths (`set_ocr_models()`) | `ensure_ocr()` |
| `table` | local slanet-plus path (`set_table_model()`) or auto-download from HF `opendatalab/PDF-Extract-Kit-1.0` (~7.8 MB) into `<cache>/models/` | `ensure_table()` (lazy — only on table regions) |

Degradation contract: model-load failures propagate as `BobineError`, but
`full_structure_page_markdown` failures are caught by the router
(`if let Ok(Some(md))`) and the page **falls back to the fast path** — a
missing layout.onnx never kills a conversion. `ModelPrecision::Fp16`
selects `*_fp16.onnx` filenames (models deferred).

---

## 5. Coordinate spaces (the one real trap)

The Rust port is *simpler* than the Python one on this front:

| Space | Used for | Source |
|---|---|---|
| **PDF points (top-left origin)** | `Rect` from `extract_chars` bboxes, formula boxes, region boxes | pdf_oxide ≥0.3 normalizes `Rect { x, y, width, height }` with y = top edge |
| **Render pixels** | rendered page images, RapidLayout outputs | `render_page(dpi)`; scale factor `dpi / 72` |

Invariants:
- `math_boxes_from_chars` returns **points** (pdf_oxide Rect directly).
- RapidLayout regions arrive in **render pixels** and are converted
  points-ward by dividing by `dpi/72` before cropping.
- `crop_image` maps points → pixels with `dpi/72`; **no y-flip needed**
  (pdf_oxide Rect is already top-left, unlike raw PDF coordinates).

---

## 6. Formula pipeline (SURGICAL)

1. **Detection**: identical heuristic to Python — math font keywords +
   unicode ranges over `extract_chars`; body fonts (`cmr`, `cmbx`)
   deliberately excluded.
2. **Line-aware merge**: chars → lines (0.8× median height baseline
   tolerance) → horizontal runs (gap ≤1.5× median height) → vertical merge
   only when lines horizontally overlap, gap fits line spacing, and merged
   box stays sane. Multi-line display equations = one crop.
3. **Recognition**: crop PNG → `TexTeller::recognize`:
   - trim white border (corner-sampled bg, threshold 15),
   - grayscale, fit within 448×448 (CatmullRom ≈ bicubic),
   - pad bottom-right, normalize `(x/255 − 0.9545467) / 0.15394445`,
   - encoder → `last_hidden_state` [1, 1024, 768],
   - greedy autoregressive decode via `decoder_model_merged.onnx` with
     KV-cache: false-branch prefill over `[bos]`, then one token per step
     with `use_cache_branch=true`. Decoder caches roll forward; encoder
     cross-attention caches are pinned from the prefill (the true branch
     emits a broken zero-batch encoder cache). ~26 ms/step flat vs
     70→244 ms full-recompute; identical greedy output,
   - BPE decode via `tokenizers` (`tokenizer.json`), bos `<s>`, eos `</s>`.
4. **Wrap + splice**: display vs inline chosen by
   `height > 1.6 × line_height ∥ width > formula_inline_max_width_pts`;
   LaTeX replaces the region's text-layer needle (whitespace-tolerant
   regex), unmatched blocks append at page end.

**P2 fallback** (`formula_layout_fallback=true`): when the text layer has
no math fonts, run RapidLayout and use `equation`-labelled regions
(`MATH_LAYOUT_LABELS`). Inline math is lost — opt-in only, same tradeoff
as Python.

---

## 7. Full structure pipeline (AUTO/ALWAYS/scanned)

`full_structure_page_markdown`:
1. Render page at `render_dpi`.
2. `layout_regions` → regions in render pixels → converted to points,
   sorted reading order (y then x).
3. Per region:
   - `table` → **text layer first** (lossless on born-digital pages);
     runs through `html_tables_to_gfm` when enabled. Scans fall back to
     **RapidTable** (SLANet-plus): the crop is OCR'd, lines are matched
     into decoded cell quads, and the resulting HTML also goes through
     `html_tables_to_gfm`. Placeholder `[table: <label>]` remains as the
     last resort.
   - math labels → crop → TexTeller → `$$…$$`.
   - `figure`/`image` → crop saved to work_dir, `![](name.png)` link.
   - else → text layer first, `ocr_lines` fallback; `title` → `## heading`.

---

## 8. Model acquisition

- **TexTeller**: downloaded automatically on first use from
  `https://huggingface.co/OleehyO/TexTeller` (`encoder_model.onnx` 344 MB,
  `decoder_model_merged.onnx` 909 MB, `tokenizer.json`) into the cache dir
  given at converter construction. `Fp16` variants would be picked up as
  `*_fp16.onnx` if present (generation deferred).
- **RapidLayout / RapidOCR**: loaded from explicit local paths set via
  `OnnxEngine::set_layout_model` / `set_ocr_models`. Label lists are read
  from the ONNX models' custom metadata key `character` (falling back to
  DocStructBench defaults / ASCII).

## 9. Version pinning philosophy

Rust dependencies are locked in `Cargo.lock`. The one external binary
contract is **onnxruntime itself**: `ort` 2.0-rc requires ≥1.19; a stale
system DLL fails at session creation with a clear `BadVersion` error.
Set `ORT_DYLIB_PATH` explicitly in CI/dev (e.g. the venv's
`onnxruntime/capi/onnxruntime.dll`).

## 10. Testing strategy

Two layers (mirrors the Python three minus fakes):

1. **Unit tests** (`#[cfg(test)]` modules, 38 tests): pure-Rust logic with
   no native deps — tables parsing, splice/ws_replace, math-font/unicode
   classifiers, TexTeller preprocessing shape/range, white-border trim,
   LetterBox preprocessing, IoU/NMS, config serde round-trip.
2. **Integration** (`tests/test_converter.rs`, 7 tests): real pdf_oxide
   against the CC BY 4.0 arXiv corpus in `tests/fixtures/` (text paper +
   scanned page), text-file conversion, config access, error paths.
   PDF tests need a modern onnxruntime (`ORT_DYLIB_PATH`).

Run: `cargo test` (45 total). Python-side smoke: `maturin develop` then
`import bobine`.

## 11. Licensing layout

- bobine_rs code: **Apache-2.0 OR MIT** (dual).
- `tests/fixtures/`: CC BY 4.0 arXiv papers (attribution in
  `tests/fixtures/SOURCES.md`).
- `legacy/`: frozen Python bobine v0.2.0, keeps its own dual license and
  vendored third-party notices.
