# bobine — Architecture

A standalone **PDF / Office / text → Markdown ingestion engine**, extracted
from OKFgraph (read-only extraction; OKFgraph keeps its own copies until it
consumes bobine — roadmap #7). This document explains how the pieces fit,
the data flows, and the non-obvious invariants.

---

## 1. Goals & non-goals

**Goals**
- Convert PDF (fast native text + optional ONNX heavy passes), Office
  (docx/xlsx/pptx), and text documents (txt/md/rst) to markdown.
- Stage extracted images into an `okf-asset://` asset store.
- Lint/normalize markdown into a graph-ready `Document` model.
- Import with **zero dependencies**; every native backend is optional and
  import-guarded; missing backends degrade to fast paths or raise clear
  `RuntimeError`s.

**Non-goals** (by design)
- No database, no embedding, no UI. The consumer owns embedding/storage
  (in OKFgraph that is `OKFRouter.import_bundle`).
- No CUDA-version coupling: ONNX runs on a plain `onnxruntime` wheel;
  CUDA is an opt-in provider list.

---

## 2. Module map

```
bobine/
├── __init__.py      public re-exports, __version__
├── config.py        ConverterConfig (dataclass) + RoutingMode (enum)
├── engine.py        OnnxRapidEngine — lazy ONNX model manager (formula/ocr/layout/table)
├── converter.py     HybridConverter — the core PDF/Office pipeline (~830 LOC)
├── tables.py        HTML table → GFM pipe-table converter
├── assets.py        okf-asset:// staging, asset_id (deterministic content hash)
├── versions.py      known-good RapidAI pins + check_rapid_versions() warning
├── documents.py     Document model, frontmatter parsing, wrap_thoughts
├── markdown.py      mordant linting (guarded; no-op E999 without mordant)
├── pipeline.py      orchestrators: convert_to_markdown / stage_images /
│                    ingest_document / convert_directory
└── _vendor/         third-party code, vendored with original licenses
    └── rapid_latex_ocr/   formula OCR (MIT (c) 2023 RapidAI, numpy-2 fixed)
```

Dependency direction: `pipeline → converter → engine → (rapidai | _vendor)`,
`pipeline → documents/markdown/assets/tables`. No cycles; `converter` never
touches the DB/graph layer.

---

## 3. Data flows

### `ingest_document(path, output_dir, …)` (pipeline.py)

1. `convert_to_markdown` dispatches by extension:
   - `.pdf` → `HybridConverter.convert_pdf` (fast path + ONNX passes per page)
   - Office exts → `office_oxide` `to_markdown()`
   - text exts → raw UTF-8 read
2. `stage_images` rewrites `![](local.png)` → `![](okf-asset://<id>)` and
   copies bytes into `output_dir/_assets/`.
3. Optional lint (`mordant`, auto-fix).
4. Returns `ConvertedDocument{md_path, md_text, image_count, page_count, lint}`.

### `HybridConverter.convert_pdf` (converter.py)

```
open PdfDocument
  for each page:
    extract embedded images (if extract_images)
    page_md = _route_page(doc, page, i, work_dir)     ← routing decision
  join pages with "---" separators
```

### Per-page routing (`_route_page`)

```
NEVER  → fast path (pdf_oxide markdown)                       [no models]
SURGICAL → scanned?  → full ONNX layout+OCR
            else     → fast + formula pass (text-layer math boxes,
                       fallback P2 layout equation regions)
AUTO   → ocr() present AND _needs_paddle(page)?
            yes → full ONNX layout+OCR
            no  → fast path
ALWAYS → full ONNX layout+OCR on every page
```

`_needs_paddle` signals: math chars in the text layer (font names /
unicode) over a threshold, or scanned-page heuristic (few chars + images).

---

## 4. OnnxRapidEngine — lazy loading & degradation

Four lazy loaders (`formula()`, `ocr()`, `layout()`, `table()`) construct
the backend on first use and cache it. Every loader is wrapped in
try/except: a constructor failure logs `⚠️ Could not load …` and returns
`None`; callers then take the fast path. This is why a born-digital paper
in `SURGICAL` loads **only** the tiny formula model, never the
OCR/layout/table stack.

All version-sensitive calls are flagged `# VERIFY`. The 3.x line returns
dataclasses (`RapidOCROutput` / `RapidLayoutOutput` / `RapidTableOutput`)
rather than tuples; `ocr_lines`/`layout_regions`/`table_html` normalize
both shapes, and `RapidOCR()` takes no kwargs in 3.x. These adapters were
verified live against rapidocr 3.9.2 / rapid_layout 1.2.1 / rapid_table
3.0.2 (Phases 5–6).

---

## 5. Coordinate spaces (the one real trap)

Two different spaces flow through the converter — never mix them:

| Space | Used for | Source |
|---|---|---|
| **PDF points** | text-layer char boxes, formula boxes, `_crop_pil` input | `TextChar.bbox` — but pdf_oxide ≥0.3 reports `(x, y, w, h)`, older/fakes report `(x0, y0, x1, y1)`. `_bbox_to_xyxy()` normalizes both. |
| **Render pixels** | RapidLayout region boxes, crops fed to OCR/table | `layout_regions()` output is in the pixel space of the image it was given |

Invariants:
- `_math_boxes_from_chars` returns **points** (used by `_crop_pil`).
- Layout regions are converted **pixels → points** (`÷ (dpi/72)`) before
  `_crop_pil` in both the full-structure path and the P2 fallback. A past
  bug passed pixels as points (≈2× oversized crops that worked by luck).
- `_crop_pil` maps points → pixels with `dpi/72` and flips y (PDF origin
  bottom-left, image top-left).

---

## 6. Formula pipeline (SURGICAL)

1. **Detection**: text-layer chars whose font name contains a TeX math
   keyword (`cmmi`, `cmsy`, `cmex`, `msam`, `msbm`, …) or unicode math
   codepoints. Body fonts (`cmr`, `cmbx`, …) are deliberately excluded —
   CMR10 is Computer Modern *Roman*, not math (a 58%-of-chars false
   positive was fixed in Phase 7).
2. **Line-aware merge**: chars → lines (baseline tolerance) → horizontal
   runs per line → vertical merge only if lines horizontally overlap, the
   gap fits ≤1.5× median line height, and the result stays ≤40 % of page
   height. Multi-line display equations = one crop; columns/prose stay
   separate.
3. **Recognition**: each crop → vendored RapidLaTeXOCR (`LatexOCR`) →
   LaTeX string.
4. **Splice**: the LaTeX replaces the region's text-layer text in the fast
   markdown (whitespace-tolerant, else appended at page end).

**P2 fallback** (`formula_layout_fallback=True`, off by default): when the
text layer has no math fonts (Word/InDesign/OCR output), run RapidLayout
and use `equation`-labelled regions (`_MATH_LAYOUT_LABELS` includes
`display_formula`/`inline_formula`/`isolate_formula` for other layout
models). Cost: pulls the layout stack into SURGICAL; layout misses inline
math — hence opt-in only.

---

## 7. Full ONNX pipeline (AUTO/ALWAYS/scanned)

`_full_structure_page_markdown`:
1. Render page at `render_dpi` → PIL.
2. `layout_regions` → regions in **render pixels** → converted to points.
3. Per region (reading order):
   - `table` → **text layer first** on born-digital pages (lossless;
     slanet HTML only when the text layer is empty — scans). Two-column
     layout models mislabel prose↔table, so text wins on digital pages.
   - math labels → crop → formula recognizer → `$$…$$`.
   - `figure`/`image` → crop staged as asset link.
   - else → text layer first, OCR fallback; `title` → `## heading`.

---

## 8. Vendoring policy

Dead-but-essential third-party code is vendored into `bobine/_vendor/`
with its original license file, rather than forked or re-pinned:

- Fixes applied in-tree (the numpy-2 `int(np.argmax(…))` crash, class-name
  alias `LatexOCR = LaTeXOCR`).
- `ruff` and coverage both exclude `_vendor/` (third-party code shouldn't
  skew our metrics or lint).
- Models (~179 MB) auto-download on first use into
  `bobine/_vendor/rapid_latex_ocr/models/` and are git-ignored.

## 9. Version pinning philosophy

RapidAI packages move fast and break APIs between minors (the `# VERIFY`
flags exist because of it). `versions.py` holds the known-good list;
`check_rapid_versions()` warns at import on drift
(`BOBINE_INGEST_ALLOW_UNPINNED=1` silences; legacy `OKFGRAPH_…` honoured).
Pins in `pyproject.toml` are exact for RapidAI packages, floors elsewhere.

## 10. Testing strategy

Three layers, each with a distinct purpose:

1. **Unit (fake pdf_oxide)** — `tests/conftest.py` fakes
   (`FakeChar/FakePage/FakePdfDocument/FakeRegion`) drive the real converter
   code through every routing, splice, and ONNX-assembly path with **no
   native deps**; engine normalization branches use injected fake engines.
   169 tests, 92 % coverage.
2. **Integration corpus** — real backends + `tests/fixtures/pdf/` (three
   CC BY 4.0 arXiv papers — text, formulas, tables, figures — plus a
   generated no-text-layer scanned page). Gated by `integration`/`slow`
   markers; self-skips when backends are missing.
3. **Parity** — OKFgraph's pure-pipeline tests run against bobine's modules
   via a sys.modules shim (35/35 green). See `docs/PARITY.md`.

## 11. Licensing layout

- bobine code: **Apache-2.0 OR MIT** (dual) — `LICENSE` pointer +
  `LICENSES/{Apache-2.0,MIT}.txt`.
- `bobine/_vendor/rapid_latex_ocr/`: MIT (c) 2023 RapidAI (its own
  `LICENSE` inside the vendored dir).
- `tests/fixtures/pdf/`: CC BY 4.0 arXiv papers (attribution +
  modification notes in `tests/fixtures/SOURCES.md`); `scanned_page.pdf`
  is generated in-repo (bobine's own license).
