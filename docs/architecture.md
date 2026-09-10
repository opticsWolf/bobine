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
├── config.rs         ConverterConfig (Routing/Render/Model/Text/Provider
│                     option groups, #[serde(flatten)] — wire format unchanged),
│                     RoutingMode, FormulaBackend, ModelPrecision,
│                     ModelQuantization
├── error.rs          BobineError enum
├── engine.rs         OnnxEngine — lazy model manager (TexTeller/RapidLayout/RapidOCR)
├── converter.rs      HybridConverter — core PDF/Office pipeline (~3.4 kLOC,
│                     split into claim_ranks / assign_glyph_owners /
│                     assign_region_lines / repair_seams / caption_block /
│                     figure_block / text_block / table_block / math_blocks)
│                     (internals generic over PdfSource)
├── pdf_source.rs     PdfSource trait: converter ↔ PDF backend seam;
│                     pdf_oxide adapter + in-memory FakePdf test fakes
├── tex_teller.rs     TexTeller ONNX: preprocess → encoder → autoregressive decode
├── rapid_layout.rs   DocLayout-YOLO: LetterBox(1024) → NMS → LayoutRegion
├── rapid_ocr.rs      PaddleOCR DBNet + CRNN: DB unclip (min-area rect +
│                     polygon offset), rotation-aware crops, CTC decode
├── rapid_table.rs    RapidTable (SLANet-plus): scanned-table → HTML
├── tables.rs         HTML <table> → GFM pipe-table converter
├── excel.rs          Excel export: convert_excel → ExcelDocument
│                     (SheetData/CellData/CellJson) + csv/json/md renderers
├── office_images.rs  Office picture staging: IR walk + package-media
│                     fallback → positional splice + gallery
├── assets.rs         okf-asset:// staging store for unreferenced images
├── documents.rs      ConvertedDocument + YAML frontmatter (ingest layer)
├── pipeline.rs       ingest_document / convert_directory / progress hooks
└── py_bindings.rs    PyO3 surface: bobine._native
python/bobine/        Python shim (__init__.py re-exports _native) + .pyi stubs
legacy/               frozen pure-Python bobine v0.2.0 (reference implementation)
```

Dependency direction: `pipeline → converter → engine → (tex_teller |
rapid_layout | rapid_ocr | rapid_table)`; `converter → tables`,
`converter → pdf_source`, `converter → excel | office_images`;
`pipeline → excel` (workbook siblings); `documents/assets` sit beside the
pipeline layer. No cycles.

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
| `.docx .xlsx .pptx .doc .xls .ppt` | `convert_office_staged` (md + staged pictures) |
| anything else | raw UTF-8 read |

> Office export (Excel csv/json, picture staging) ships — see
> `IMPLEMENTATION_PLAN_office.md` and §12.

### Office conversion (`convert_office_staged`)

```
Document::open(path)                       ← office_oxide, magic-byte sniffed
md = doc.to_markdown()                     ← upstream rendering (quirks logged
                                              in IMPLEMENTATION_PLAN_office.md)
if !extract_images: return md              ← knob respected, md untouched
images = collect_office_images(doc.to_ir())
if images.empty(): images = collect_package_images(path)
return splice_office_images(md, images,    ← stage + positional rewrite +
                            <work>/<image_output_dir>/office/, stem)   gallery
```

Pure `HybridConverter::convert_office(path)` (associated function — no
state, config, or models) skips staging and returns the upstream markdown
as-is; the dispatcher always stages. Staged links are work-dir-relative
(`assets/office/img_<hash>.png`), so the ingest `stage_images` pass
promotes them to `okf-asset://` with no pipeline changes (it gained one
resolution candidate: work-dir-relative targets).

### Excel export (`convert_excel`)

```
Document::open → as_xlsx / as_xls
  per sheet: grid of CellData { text (formatted display),
                                 value (typed: int/float/bool/text/null),
                                 formula (=… cached, never evaluated) }
  renderers: sheets_to_markdown (## {sheet} + GFM) · sheets_to_csv
             (RFC-4180, empty sheets skipped) · excel_to_json
pipeline: ingest writes <stem>.<sheet>.csv + <stem>.json siblings,
          recorded in ConvertedDocument.data_files
```

Integer-valued floats stay integers in JSON (`12`, not `12.0`);
dates/percents render as display text; errors keep text with `null` value;
interior empty rows are preserved (row alignment matters for formulas).

---

## 4. OnnxEngine — lazy loading & degradation

Three lazy slots, each loaded on first use and cached:

| Slot | Model source | Loaded by |
|---|---|---|
| `tex_teller` | HuggingFace via `hf-hub` (blocking): default **Int8** from `Ji-Ha/TexTeller3-ONNX-dynamic` into `<cache>/texteller_int8/`; `ModelQuantization::Fp32` uses `OleehyO/TexTeller` | `ensure_tex_teller()` |
| `layout` | HF `wybxc/DocLayout-YOLO-DocStructBench-onnx` (auto-download, probe-first) or local path (`set_layout_model()`) | `ensure_layout()` |
| `ocr` | HF `SWHL/RapidOCR` PP-OCRv4 det+rec (auto-download, probe-first) or local paths (`set_ocr_models()`) | `ensure_ocr()` |
| `table` | local slanet-plus path (`set_table_model()`) or auto-download from HF `opendatalab/PDF-Extract-Kit-1.0` (~7.8 MB) into `<cache>/models/` | `ensure_table()` (lazy — only on table regions) |

Degradation contract: model-load failures propagate as `BobineError`, but
`full_structure_page_markdown` failures are caught by the router
(`if let Ok(Some(md))`) and the page **falls back to the fast path** — a
missing layout.onnx never kills a conversion. `ModelPrecision::Fp16`
selects `*_fp16.onnx` filenames (models deferred).

Provider resolution (v0.4.9+): `ConverterConfig.ort_providers` is the base
list for TexTeller sessions. Every heavy session has a per-slot override —
`encoder_ort_providers` / `decoder_ort_providers` (TexTeller),
`layout_ort_providers`, `ocr_ort_providers`, `table_ort_providers`; `None`
selects the default, which encodes the measured ledger:

| Slot | Default resolution | Why |
|---|---|---|
| layout, ocr | **auto-GPU** — CUDAExecutionProvider prepended when the loaded dylib registers it (once-locked probe) | measured 12.3x / 3.6x on CUDA |
| table | **CPU-pinned** | SLANet measures 2-9x slower on CUDA (graph fragments across devices) |
| tex_teller | base `ort_providers` | Int8+CUDA guarded with a warning; Fp32+CUDA opt-in via the base list |

On CPU-only ONNX Runtime builds the CUDA probe is false and every slot
resolves to plain CPU — zero cost, no config needed either way.

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
   - normalize `(x/255 − 0.9545467) / 0.15394445`, then pad bottom-right —
     the pad fill is 0.0 in *normalized* space (= raw ~243, background
     white), matching upstream's Normalize-before-pad order; padding with
     raw black measurably degrades recognition (e.g. `\oint` misreads),
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
4. **Seam repair**: glyphs owned by no region are clustered into lines and
   re-assigned to the nearest region whose padded box contains them;
   only truly homeless lines fall through to a trailing block.
5. **Post-passes** (all modes): `promote_headings` maps scholarly section
   patterns (`I. X`, `3.1 Y`, `3.1.1 Z`, `A. W`) to `##`–`####`, guarded
   against prose/lists/axis labels; `promote_title` promotes the document
   title to `#` (first short block, or the first title-like heading on
   page 1).

---

## 8. Model acquisition

- **TexTeller**: downloaded automatically on first use into the cache dir
  given at converter construction. Default (`ModelQuantization::Int8`):
  `https://huggingface.co/Ji-Ha/TexTeller3-ONNX-dynamic`
  (`onnx/encoder_model.onnx` ~90 MB int8, `onnx/decoder_model_merged.onnx`
  ~909 MB fp32-with-cache-interface, `tokenizer.json`) into
  `<cache>/texteller_int8/`. Fp32 fallback:
  `https://huggingface.co/OleehyO/TexTeller` (`encoder_model.onnx` 344 MB,
  `decoder_model_merged.onnx` 909 MB, `tokenizer.json`). Tokenizer and
  weights are paired per variant in distinct cache namespaces. `Fp16`
  variants would be picked up as `*_fp16.onnx` if present (generation
  deferred).
- **RapidLayout / RapidOCR**: downloaded automatically on first use unless
  explicit local paths are set via `OnnxEngine::set_layout_model` /
  `set_ocr_models`. Layout: `wybxc/DocLayout-YOLO-DocStructBench-onnx`
  (`doclayout_yolo_docstructbench_imgsz1024.onnx`, 72 MB - a community ONNX
  conversion of the official DocStructBench weights, whose repo ships .pt
  only). OCR: `SWHL/RapidOCR` `PP-OCRv4/ch_PP-OCRv4_{det,rec}_infer.onnx`
  (~16 MB total). Label lists / charsets are read from the ONNX models'
  custom metadata key `character` (falling back to DocStructBench defaults /
  ASCII); the rec charset is aligned against the model's actual class count
  at first decode, since exports vary in whether they include the leading
  CTC blank and trailing space classes.

## 9. Version pinning philosophy

Rust dependencies are locked in `Cargo.lock` (update deliberately with
`cargo update -p <crate>`, never blindly — the pdf_oxide 0.3.77→0.3.78
roll moved whole-corpus golden output and was reviewed file-by-file before
landing as v0.5.6). Current oxide pins: `office_oxide 0.1.10`,
`pdf_oxide 0.3.78`. The one external binary contract is **onnxruntime
itself**: `ort` 2.0-rc requires ≥1.19; a stale system DLL fails at session
creation with a clear `BadVersion` error. Set `ORT_DYLIB_PATH` explicitly
in CI/dev (e.g. the venv's `onnxruntime/capi/onnxruntime.dll`).

## 10. Testing strategy

Three layers:

1. **Unit tests** (`#[cfg(test)]` modules, 128 lib tests): pure logic —
   tables parsing, splice/ws_replace, math-font/unicode classifiers,
   TexTeller preprocessing shape/range, white-border trim, LetterBox,
   IoU/NMS, config serde round-trip, Excel renderers (GFM/CSV/JSON schema),
   office splice/collect. No models, no files.
2. **Integration** (`tests/`): real fixtures, no mocks —
   `test_office` (10: generated OOXML fixtures, ordered-content + staged
   pictures + goldens), `test_excel` (6: typed values, csv round-trip,
   ingest siblings), `test_golden` (2: PDF corpus + figures goldens — the
   tripwire for upstream pdf_oxide changes), `test_rapid_table` (model
   load + recognize), `test_converter` (7: config, text files, error
   paths, PDF fast paths incl. the ~2 min full-paper AUTO run).
3. **Python smoke**: `maturin develop`, then exercise `convert_excel`,
   `convert`, and `ingest_document` end-to-end (Phase 4 protocol).

Golden policy: goldens pin *intentional* output. Regenerate only via
`BOBINE_UPDATE_GOLDENS=1` after reviewing the diff (`tests/golden/` for
PDF, `tests/golden/office/` for OOXML — deterministic thanks to
content-hash staged links). Fixture provenance lives in
`tests/fixtures/SOURCES.md` (CC BY 4.0 arXiv trims + generated scanned
page + generated OOXML fixtures via `examples/gen_office_fixtures.rs`).

Run: `cargo test` (needs `ORT_DYLIB_PATH` — without a resolvable dylib
the test binary aborts). CI (`.github/workflows/ci.yml`) runs the full
locked suite on ubuntu with a CPU onnxruntime. Legacy Python keeps its own
frozen tests under `legacy/tests/`.

## 11. Licensing layout

- bobine_rs code: **Apache-2.0 OR MIT** (dual).
- `tests/fixtures/`: CC BY 4.0 arXiv papers (attribution in
  `tests/fixtures/SOURCES.md`).
- `legacy/`: frozen Python bobine v0.2.0, keeps its own dual license and
  vendored third-party notices.

---

## 12. Office export subsystem (v0.5.x)

Full background: `IMPLEMENTATION_PLAN_office.md`. The subsystem has three
parts with deliberately different trust levels:

- **`convert_office`** — thin delegation (`Document::open` +
  `to_markdown`). All rendering intelligence is office_oxide's; bobine
  pins *must-hold* behavior in `test_office` (GFM tables, sheet/slide
  `##` boundaries, typed cells, alt text) and logs the rest as upstream
  gaps (footnote bodies, hyperlink URLs, formula cached values, pptx
  bullets/markers, pptx TSV tables). No in-tree workarounds — upstream
  fixes with minimal repros instead.
- **`office_images`** — bobine-owned staging. Two sources, one contract:
  the IR walk (document order, alt text, decorative + link-only skipped,
  table/textbox/note nesting) wins; when the IR carries nothing (pptx/xlsx
  drawings never reach it — verified), the `*/media/*` package scan fills
  in (filename-stem alt). Positional pairing against md links, surplus
  pictures gallery-appended — the same shape as the PDF unreferenced-figure
  flow, so `ingest_document` promotes both identically.
- **`excel`** — bobine-owned data model over office_oxide cell access
  (`format_cell_value` for xlsx, `as_text` for xls). Formulas are cached
  values only, never evaluated — a load-bearing decision for JSON consumers.

Fixture strategy differs from PDF by necessity: no Office corpus can be
redistributed, so `examples/gen_office_fixtures.rs` *generates* fixtures
from office_oxide's own `create` API (IR→OOXML round-trip by construction).
Legacy doc/xls/ppt have no writers — no fixtures until real samples arrive;
the harness and the IR walk are format-agnostic so coverage lands with
zero code changes.

## 13. Error-handling philosophy

- `BobineError` is the single error type; upstream errors cross the seam
  as strings (`OfficeOxide(String)`, `PdfOxide(String)`) — coarse but
  uniform, and never panics on user files (empty-docx test pins this).
- **Degradation beats failure**: per-page fast-path fallback (missing
  models), per-sheet CSV skips (empty sheets), per-sibling warn-and-
  continue (Excel siblings never abort an ingest), provider
  auto-degradation (CUDA requested, CPU dylib loaded → CPU silently).
- Ingest siblings and staging are best-effort with `warn!`; conversion
  results are exact or errored, never half-written (md written before
  staging, lint last).

## 14. Python binding layer

`py_bindings.rs` (feature `extension-module`) exposes `bobine._native`;
`python/bobine/__init__.py` re-exports a curated surface with `.pyi` stubs
as the documented contract. Rules:

- **Additive only** — new surface (`convert_excel`, `ExcelDocument`,
  `ConvertedDocument.data_files`) extends; existing signatures never break.
- **Thin wrappers** — no logic in bindings; every `#[pyfunction]` is a
  direct call into `src/` plus error mapping (`PyRuntimeError`).
- **Flat kwargs** — `PyConverterConfig` takes the same flat field names as
  the pre-grouping Rust struct (the `#[serde(flatten)]` grouping is a
  Rust-side organization; the Python surface never moved).
- The default (non-extension) build is Python-free — enforced in CI
  (`cargo check` without the feature) and at publish time.
