# bobine — Consolidated Implementation Plan

**Status:** v0.1.0 · dual-licensed (Apache-2.0 OR MIT) · `https://github.com/opticsWolf/bobine`
**Last updated:** 2026-08-09 · latest commit pending (docs: quickref + architecture)

**Docs:** [quickref.md](quickref.md) · [architecture.md](architecture.md) · [PARITY.md](PARITY.md)

---

## 1. Goal

Extract the **document ingestion pipeline** from the OKFgraph project
(`okfgraph/ingest/` + the graph-agnostic parts of `okfgraph/components/ingest.py`)
into a **standalone, dependency-light module** (`bobine`) that:

- converts **PDF** (pdf_oxide fast path + RapidAI/ONNX heavy passes),
  **Office** (docx/xlsx/pptx via office_oxide), and **text-type** documents
  (txt/md/rst) to Markdown;
- stages extracted images into an `okf-asset://` asset store;
- lints and normalizes markdown/text documents into a graph-ready
  `Document` model;
- never touches a database, embedding model, or UI — consumers own that.

OKFgraph itself is **not modified**; it keeps importing its own
`okfgraph.ingest.*`.

---

## 2. Status Summary

| Area | Status |
|---|---|
| PDF/Office → Markdown pipeline | ✅ extracted (`config`, `engine`, `converter`, `tables`) |
| Image staging (`okf-asset://`) | ✅ extracted (`assets`) |
| Version pinning / runtime check | ✅ extracted (`versions`), pins refreshed to resolvable set |
| Text-type document model + lint | ✅ new (`documents`, `markdown`) |
| Orchestration (single + batch) | ✅ new (`pipeline`) |
| Formula OCR (SURGICAL mode) | ✅ **vendored** `rapid_latex_ocr` with numpy-2 fix; verified live on numpy 2.5.1 / py3.13 / onnxruntime 1.28 |
| Dependency tree | ✅ modernised + `uv.lock` (single numpy-2 lineage); `office_oxide` finally declared |
| Unit tests (fake pdf_oxide, no deps) | ✅ **169 passed**, 11 integration |
| Coverage | ✅ **92%** (converter 91%, engine 93%, pipeline 97%; `_vendor` excluded) |
| Lint / format (ruff) | ✅ clean |
| CI (GitHub Actions) | ✅ core matrix 3.10–3.13 + integration job |
| Git / remote | ✅ `main` on GitHub, clean tree |
| Real-backend verification | ✅ **done (2026-08-09)**: pdf_oxide 0.3.77 + office_oxide + all four ONNX stacks installed & run end-to-end (SURGICAL + ALWAYS); engine adapters updated for rapidocr/layout/table 3.x dataclass returns |
| Test-PDF corpus | ✅ **done (2026-08-09)**: 3 CC BY 4.0 arXiv papers (trimmed, attributed) + generated scanned page; corpus caught 2 real bugs (ndarray `or` crash, rapid_table ocr_results format) |

---

## 3. Repository Layout

```
bobine/
├── pyproject.toml            # deps, extras, pytest/coverage/ruff config
├── uv.lock                   # universal lockfile (uv)
├── README.md                 # usage, output contract, testing
├── LICENSE                   # dual-license pointer (choose either)
├── LICENSES/                 # Apache-2.0.txt, MIT.txt
├── .github/workflows/ci.yml  # lint+unit (3.10–3.13), integration job
├── bobine/
│   ├── __init__.py           # public API + __version__
│   ├── config.py             # ConverterConfig, RoutingMode
│   ├── engine.py             # OnnxRapidEngine (lazy ONNX model manager)
│   ├── converter.py          # HybridConverter (core PDF/Office pipeline)
│   ├── tables.py             # HTML table → GFM pipe-table converter
│   ├── assets.py             # okf-asset:// staging, asset_id
│   ├── versions.py           # RapidAI version pins + runtime check
│   ├── documents.py          # Document model, frontmatter, wrap_thoughts
│   ├── markdown.py           # mordant linting (guarded)
│   ├── pipeline.py           # convert_to_markdown / stage_images /
│   │                         #   ingest_document / convert_directory
│   └── _vendor/              # third-party code, vendored with licenses
│       └── rapid_latex_ocr/  # formula OCR (MIT (c) 2023 RapidAI, numpy-2 fixed)
└── tests/                    # 13 files, 116 unit + 6 integration tests
    └── fixtures/             # formula_sample.png etc.
```

### Extras (all optional — the package imports with zero dependencies)

| Extra | Provides |
|---|---|
| *(core)* | `Pillow` |
| `[pdf-ingest]` | `pdf_oxide`, `office_oxide`, `rapidocr==3.9.2`, `rapid_layout==1.2.1`, `rapid_table==3.0.2`, `numpy>=2` |
| `[formula]` | vendored formula OCR runtime: `onnxruntime`, `tokenizers`, `opencv-python`, `chardet`, `requests`, `pyyaml` (models ~179 MB auto-download on first use) |
| `[markdown]` | `mordant`, `python-frontmatter`, `pyyaml` |
| `[dev]` | `pytest` |

---

## 4. Architecture & Design Decisions

1. **Graph-agnostic output contract.** `ingest_document()` produces
   `<stem>.md` (linted, `okf-asset://` links) + `_assets/<id>.<ext>`. A
   consumer (e.g. OKFgraph's `import_bundle`) does embedding/storage.
2. **Everything optional.** Native backends are import-guarded exactly as in
   OKFgraph; missing backends degrade to fast paths or raise clear
   `RuntimeError`s. Frontmatter falls back to plain-text reading without
   `python-frontmatter`; linting to a no-op `E999` diagnostic without
   `mordant`.
3. **Routing modes.** `NEVER` / `AUTO` / `SURGICAL` / `ALWAYS` control when
   ONNX models load — a born-digital PDF loads no models at all.
4. **Testable without native deps.** pdf_oxide's API is duck-typed
   (`page.markdown()`, `page.chars`/`bbox`/`font_name`, `doc.within()`,
   `doc.extract_image_bytes()`); `conftest.py` fakes drive every routing,
   splicing, and ONNX-assembly path in the real converter code.
5. **Version pinning.** `check_rapid_versions()` warns on drift from the
   known-good RapidAI list at import time (`BOBINE_INGEST_ALLOW_UNPINNED=1`
   silences; legacy `OKFGRAPH_INGEST_ALLOW_UNPINNED` honoured).
6. **Vendoring over forking.** Dead-but-essential third-party code is
   **vendored** (`bobine/_vendor/`) with the original license preserved and
   patches applied in-tree (see Phase 4), rather than maintained as a
   separate fork. `ruff` and coverage both exclude `_vendor/`.
7. **Dual license.** `Apache-2.0 OR MIT` (SPDX) — pointer `LICENSE` +
   full texts in `LICENSES/`, mirroring OKFgraph's convention.

---

## 5. Implementation Phases

### Phase 0 — Analysis (done)
- Mapped OKFgraph's ingestion surface: `okfgraph/ingest/` (6 modules,
  1333 LOC), consumers (`cli.py`, `components/ingest.py`, `mcp_server.py`,
  examples), duplicated staging logic, and the graph-agnostic text pipeline
  (mordant lint, frontmatter, thought wrapper).
- Concluded the pipeline was ~fully decoupled already — only imports,
  docstrings, and one env var tied it to okfgraph.

### Phase 1 — Extraction (done)
- Copied `config/engine/converter/tables/assets/versions` verbatim; rewrote
  imports (`okfgraph.ingest.*` → `bobine.*`), renamed the env var, fixed the
  build backend to standard `setuptools.build_meta`.
- New `documents.py`, `markdown.py`, `pipeline.py`.
- Ported OKFgraph's pure-pipeline tests; added text/pipeline tests.

### Phase 2 — Testing framework (done)
- `conftest.py`: `FakeChar/FakePage/FakePdfDocument/FakeRegion`, PNG fixture.
- `test_converter_flow.py`: 35 tests over routing signals, formula-box
  merging, splicing, code blocks, image extraction, ONNX page assembly
  (layout→OCR/table/formula/figure) via injected fake engines.
- `test_integration.py`: real-backend tests (synthetic PDF, page count,
  minimal-zip .docx) behind the `integration` marker + `importorskip`.
- Markers (`integration`, `slow`), coverage config, ruff config in
  `pyproject.toml`; **lint+format clean; 78% coverage**.
- `.github/workflows/ci.yml`: core matrix (3.10–3.13) + integration job.

### Phase 3 — Repository & release hygiene (done)
- Git repo created, `main` pushed to GitHub; GitHub's auto-initialized
  `LICENSE` (MIT) merged and upgraded to dual Apache-2.0 OR MIT.
- Version pinned to **0.1.0** (pyproject + `__version__` + dist metadata).

### Phase 4 — Dependency modernisation & formula OCR (done, `709d31a`)

### Phase 5 — Real-backend verification (done, `2026-08-09`)

### Phase 6 — Test-PDF corpus (done, `2026-08-09`)

### Phase 7 — Formula-box detection fix (done, `…`)
- Corpus exposed roadmap #2a's root cause as **two real bugs** (not a
  tuning issue):
  1. `"cmr10"` was in `_MATH_FONT_KEYWORDS` — CMR10 is Computer Modern
     **Roman**, the LaTeX body font; 3,103 of 5,308 chars (58%) on a real
     paper page were flagged as math. Removed.
  2. pdf_oxide >=0.3 reports `TextChar.bbox` as `(x, y, w, h)`, the merge
     assumed `(x0, y0, x1, y1)` — width/height were compared against x/y,
     so `_overlaps()` was true for nearly every pair → one page-spanning
     box. Added `_bbox_to_xyxy()` (format-agnostic) used by
     `_math_boxes_from_chars` + `_region_text`.
- Result on real papers: 1 page-spanning box → **tight equation-sized
  boxes** (display equations ~205×31pt, stacked fractions 164×136pt);
  prose-only page 1 → 0 boxes; formulas now splice **in place** (mid-
  document, not appended at page end); 30 recognized formula blocks across
  5 pages of splitting_methods.pdf in ~60s.
- Tuning: `min_formula_math_chars` default 3 → 5 (skips tiny fragments;
  63 → 44 boxes on the 4-page physics fixture).
- Regression tests: cmr10/cmbx12 body text NOT math; (x,y,w,h) + legacy
  bbox formats both yield tight boxes; full suite **126 passed**.
- Remaining quality gap (roadmap #2a): multi-line display equations are
  split per line (VGAP=8pt < 11pt line pitch) → per-line crops reduce
  recognition fidelity; layout-equation fallback (P2) remains the
  font-agnostic upgrade path.

### Phase 8 — Line-aware merge (P1) + layout-equation fallback (P2) (done, `…`)
- **P1 — line-aware merge** in `_math_boxes_from_chars`: flagged chars are
  grouped into lines (baseline tolerance), merged horizontally per line,
  then adjacent lines merge vertically ONLY if they horizontally overlap,
  the gap fits ≤1.5× median line height, and the result stays ≤40% of page
  height. A 2-line display equation is now ONE box (1 crop, 1 recognition);
  separate equations, columns and prose stay separate. Verified on
  splitting_methods.pdf.
- **P2 — optional layout fallback** (off by default):
  `ConverterConfig.formula_layout_fallback=False`; when enabled and the
  text-layer detector finds nothing, `_layout_equation_boxes()` runs
  RapidLayout and uses math-labelled regions (`_MATH_LAYOUT_LABELS` =
  equation/display_formula/inline_formula/isolate_formula/formula).
  Empirically: cdla detects ~0 equation regions on the physics/math corpus
  (inline math is text to it), so P2 is a genuine edge-case path for
  text-layer-hostile PDFs, not a replacement.
- **Latent scaling bug fixed**: RapidLayout returns boxes in render-PIXEL
  space; `_crop_pil` expects points. The ALWAYS full-structure path passed
  pixels through as points (≈2× oversized crops that only worked by luck);
  both the full-structure path and P2 now convert pixels→points
  (÷(dpi/72)). Corpus ALWAYS test now exercises the corrected geometry at
  the design-default 300dpi (tests had been under-driving it at 150dpi).
- **New quality finding (roadmap)**: slanet-plus collapses the two-column
  trust_ml table to a single row (1 tr × 2 td) — a structure-recognition
  limitation on layout boxes spanning both columns, not a regression; the
  OCR'd content still survives (GFM or raw HTML).
- Tests: 6 new unit tests (line-aware merge ×3, P2 pixel→point + label
  filter + surgical on/off wiring); full suite **133 passed**.

### Phase 9 — Table quality: text-layer-first on born-digital pages (done, `dc394b3`)

### Phase 10 — Coverage 83% → 92% (done, `…`)
- Roadmap #3 closed: TOTAL **92%** (target ≥85%); converter 91%, engine 93%,
  pipeline 97%, versions 93%, tables/markdown/assets ≥92%.
- 33 new tests, all via fakes/monkeypatch — no new runtime deps:
  - engine: 4 lazy-loader failure branches (constructor raises → log +
    degrade), recognize_formula failure/blank/no-engine, legacy OCR short
    rows (score defaults 1.0), table no-engine/empty-htmls.
  - converter: spans-based math signal, `_is_scanned`/`_needs_paddle`
    exception paths, degenerate `_crop_pil`, `_ws_replace` no-match +
    tolerant match, `_obj_to_pil`/`_render_page_to_png` branch matrix,
    `_extract_page_images` attr/dict/no-data/error paths, np-None guards,
    render-None paths, `convert_office` missing/success.
  - flow/pipeline/versions: AUTO-no-paddle routing, page_count fallback,
    office backend missing in `convert_to_markdown`, unsupported extension,
    directory non-file skip + missing-dir, version parse edge cases,
    drift-warning without logging.
- Full suite **169 passed**.
- **Diagnosis of roadmap #2b** (slanet 1×2 collapse on trust_ml): the
  layout model was the problem, not slanet — `layout_cdla` labels a
  two-column PROSE block as `table` and the real results table as `text`.
  Tried `pp_doc_layoutv3` (124 MB, newer): same mislabels. So no model swap
  fixes it.
- **Fix**: in `_full_structure_page_markdown`, prefer the **text layer** for
  `text` and `table` regions on born-digital pages (lossless); OCR / slanet
  run only when the text layer is empty (scans). Result on trust_ml ALWAYS:
  real table text lossless ("Model Clean Acc.↑ Misleading Acc.↑ Seed 1.8
  93.3±0.8…"), no garbage HTML, 26 assets staged, **16.7s → 2.3s**.
- Tests: 3 new unit tests (text layer preferred over OCR; degenerate table
  output bypassed when text layer has content; slanet kept when no text
  layer). Full suite **136 passed**; corpus suite ~1 min faster.
- **Sources**: 3 arXiv papers verified **CC BY 4.0** on their abstract pages
  (the search-UI license filter is leaky — verification is per-paper):
  `2608.06342` solitons (physics, equation-dense), `2608.05540` splitting
  methods (math), `2608.06377` trust-ML (big tables + figures).
- **License compliance**: CC BY 4.0 requires attribution + modification
  notice → `tests/fixtures/SOURCES.md` records title/authors/arXiv ID/license
  and the page ranges we trimmed. Full PDFs stay **git-ignored** in
  `tests/fixtures/full_pdfs/` for local tests (user requirement).
- **Scanned page**: reportlab-generated raster page with **no text layer**
  (`tests/fixtures/generate_corpus.py`, deterministic, committed) — exercises
  the `_is_scanned` → layout+OCR path that born-digital arXiv PDFs cannot.
- **Empirical results on real documents** (all four ONNX stacks verified):
  - NEVER: full text extraction (titles, 20k+ chars/page-set) ✓
  - SURGICAL: **real formula detection + LaTeX recognition** per page ✓
    (splice lands at page end when the math-box merge spans the page —
    in-place replacement needs finer math-box detection; quality note)
  - scanned SURGICAL: OCR reads the raster back ✓ (741 chars, real words)
  - ALWAYS on the ML paper: layout → rapid_table HTML ✓ + figures staged
    as `okf-asset://` (26 assets) ✓
- **Bugs the corpus caught & fixed**: `getattr(.., 'boxes', None) or []`
  crashes on real OCR results (ndarray truthiness — numpy-2); rapid_table
  3.x needs per-image `[boxes_array, txts_tuple, scores_tuple]`; unguarded
  `crop=None` fed NoneType to OCR/formula saves.
- Tests: `tests/test_pdf_corpus.py` (5 integration+slow), full suite
  **122 passed**; reportlab added to `[dev]` (generator-only).
- Installed `[pdf-ingest]` on py3.13/numpy 2.5.1: `rapidocr==3.9.2`, `rapid_layout==1.2.1`, `rapid_table==3.0.2`, `pdf_oxide==0.3.77`, `office_oxide==0.1.8` — all 12 packages resolved and installed cleanly.
- **Integration suite now green end-to-end**: 6/6 (formula recognition ×2, pdf_oxide NEVER-mode conversion, page count, missing-file, office docx) + full suite **116 passed**.
- **API drift found & fixed** (the `# VERIFY` gamble paid off):
  - `RapidOCR()` takes **no kwargs** in 3.x — dropped `use_angle_cls`/`*_use_cuda` (defaults = onnxruntime/CPU/use_cls=True).
  - `rapid_layout` 1.2.1 and `rapid_table` 3.0.2 return **dataclasses** (`RapidLayoutOutput`, `RapidTableOutput`), not tuples — added normalization branches (with legacy fallbacks).
  - pdf_oxide 0.3.77 is an API **rewrite** (render/within/extract_chars on the doc; Page proxies `markdown`/`render`/`chars`) — the converter's duck-typed surface still lands: `page.render(dpi=…) → bytes → PIL` verified.
- **ALWAYS-mode smoke test**: all four ONNX models download + load + run on a rendered page (rapidocr PP-OCRv6 det/cls/rec, layout_cdla); OCR det ran (empty on the synthetic page — expected).
- Unit coverage for the new dataclass branches added (engine 43% → **82%**; total 78% → **83%**).
- Also fixed: tests that clobbered `bobine.converter.PdfDocument` to `None` (monkeypatch now); the missing-backend unit test is env-agnostic (real pdf_oxide raises `OSError` on an invalid PDF).
- **Dep-tree audit** found the tree was broken: `rapidocr==1.5.2` and
  `rapid_latex_ocr==1.0.13` **don't exist on PyPI** (install of
  `[pdf-ingest]` would fail), and `office_oxide` (imported by the converter)
  was **undeclared**.
- **Modern pins** (verified to resolve): `rapidocr==3.9.2`,
  `rapid_layout==1.2.1`, `rapid_table==3.0.2`, `pdf_oxide>=0.2.1`,
  `office_oxide>=0.1.8`, `numpy>=2`. `uv.lock` generated (single numpy-2
  lineage — the old numpy-1 `[formula]` fork is gone).
- **Successor research** (no maintained RapidAI successor exists):
  upstream dormant since 2024-11; all 41 forks 0-star; `RapidLatex` is a
  translation tool, not OCR; `TexTeller` (80M pairs, Apache-2.0) is the
  ecosystem quality upgrade but pulls the torch stack (~2.5 GB).
- **Vendored `rapid_latex_ocr`** into `bobine/_vendor/` (MIT (c) 2023
  RapidAI, license preserved) and fixed the **numpy-2 breaking bug** upstream
  never addressed: `int(np.argmax(...))` on a `(1, 21)` ONNX output →
  `.item()`. Verified live: formula crop → LaTeX in 0.49 s on
  numpy 2.5.1 / py3.13 / onnxruntime 1.28. Both `LaTeXOCR` (upstream) and
  legacy `LatexOCR` (what bobine's engine imported) names exposed.
- `[formula]` extra now declares the vendored runtime deps; models (~179 MB)
  auto-download on first use from the (still-alive) RapidAI release and are
  git-ignored.

---

## 6. Open Work — Prioritized Roadmap

| # | Item | Effort | Why |
|---|---|---|---|
| 1 | ~~PDF/Office runtime verification~~ ✅ **done (2026-08-09)** — pins installed, integration suite green, drift fixed; see Phase 5 | — | — |
| 2 | ~~Test-PDF corpus~~ ✅ **done (2026-08-09)** — 3 CC BY arXiv papers + generated scanned page; see Phase 6 | — | — |
| 2a | **Formula splice placement** ✅ **done (2026-08-09)** — root cause was two real bugs (cmr10 body-font false positive; bbox `(x,y,w,h)` vs `(x0,y0,x1,y1)` drift), fixed + tuned; formulas splice in place. **P1 line-aware merge** (multi-line equations = 1 box) and **P2 layout fallback** (`formula_layout_fallback`, off by default) implemented in Phase 8 | — | — |
| 2b | **Table structure quality** ✅ **done (2026-08-09)** — root cause: layout model mislabels two-column pages (prose→table, real table→text); v3 model no better; fixed via text-layer-first on born-digital pages (lossless, 7× faster). Slanet HTML still used for scans | — | — |
| 3 | ~~Raise coverage~~ ✅ **done (2026-08-09)** — **92%** total (converter 91%, engine 93%, pipeline 97%); 33 new fake/monkeypatch tests; see Phase 10 | — | — |
| 4 | ~~Parity check~~ ✅ **done (2026-08-09)** — OKFgraph's 35 pure-pipeline tests pass against bobine via shim; all code drift intentional; see `docs/PARITY.md` | — | — |
| 5 | **`slow` GPU job** in CI (onnxruntime CUDA) — optional | L | GPU provider path (`ort_providers`) untested |
| 6 | **PyPI publish** (0.1.0 or 0.2.0): `uv build`/twine, long description, classifiers | S | Distribution |
| 7 | **Consume from OKFgraph** (optional follow-up, per user constraint OKFgraph stays untouched for now): re-export shim or refactor of `cli.py`/`components/ingest.py` | M | Remove duplication, single source of truth |
| 8 | ~~Converter label fix~~ ✅ **done (2026-08-09, Phase 8)** — `_MATH_LAYOUT_LABELS` = equation/display_formula/inline_formula/isolate_formula/formula, used by both the full-structure path and the P2 fallback (was `("formula", "equation", "isolate_formula")`) | — | — |
| 9 | **Formula accuracy upgrade (optional)**: TexTeller (80M pairs) or pix2tex-ONNX as an alternative recognizer behind `OnnxRapidEngine.recognize_formula` | M | Vendored model is the 100K-pair accuracy ceiling |

**Legend:** S = < 1 day · M = 2–3 days · L = 1+ week

---

## 7. How to Run

```bash
cd D:/User/Documents/Python/bobine
uv venv .venv && uv pip install --python .venv/Scripts/python.exe -e ".[dev,markdown]"

# unit suite (no native backends needed)
.venv/Scripts/python.exe -m pytest

# integration suite (formula: needs bobine[formula]; PDF/Office: bobine[pdf-ingest])
.venv/Scripts/python.exe -m pytest -m integration

# coverage + lint + lockfile
.venv/Scripts/python.exe -m pytest --cov=bobine --cov-report=term-missing
.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .
uv lock --check
```

---

## 8. Acceptance Criteria (definition of done for v1.0)

- [x] **169+** unit tests pass on a bare install (no optional deps)
- [ ] Integration suite green on a runner with `bobine[pdf-ingest]` + `bobine[formula]` installed (**demonstrated locally 2026-08-09**)
- [x] Coverage ≥ 85% on `bobine/converter.py` + `bobine/pipeline.py` (**92%** total, converter 91%, pipeline 97%)
- [ ] CI green on Python 3.10–3.13 (lint, format, unit, coverage)
- [x] Test-PDF corpus committed under `tests/fixtures/` — 3 CC BY 4.0 arXiv
      papers (trimmed, `SOURCES.md` attribution) + generated scanned page
- [ ] RapidAI version pins (`versions.py`) match `pyproject.toml` **and** pass
      a real-install runtime smoke test
- [ ] `display_formula` / `inline_formula` routed to the formula recognizer
- [ ] Dual license metadata correct (`Apache-2.0 OR MIT`), vendored MIT notice
      intact, buildable wheel (`uv build` + `pip install` the wheel)
