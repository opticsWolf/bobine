# bobine — Consolidated Implementation Plan

**Status:** v0.1.0 · dual-licensed (Apache-2.0 OR MIT) · `https://github.com/opticsWolf/bobine`
**Last updated:** 2026-08-09 · latest commit `5154ba2` (real-backend verification, API-drift fixes)

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
| Unit tests (fake pdf_oxide, no deps) | ✅ **116 passed**, 6 integration |
| Coverage | ✅ **83%** (engine 82% — was 43%, converter 76%, assets 92%; `_vendor` excluded) |
| Lint / format (ruff) | ✅ clean |
| CI (GitHub Actions) | ✅ core matrix 3.10–3.13 + integration job |
| Git / remote | ✅ `main` on GitHub, clean tree |
| Real-backend verification | ✅ **done (2026-08-09)**: pdf_oxide 0.3.77 + office_oxide + all four ONNX stacks installed & run end-to-end (SURGICAL + ALWAYS); engine adapters updated for rapidocr/layout/table 3.x dataclass returns |
| Table/formula-page recognition corpus | ⬜ pending (roadmap #2) |

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
| 2 | **Test-PDF corpus** (reportlab-generated at test time): digital text, math formulas, GFM tables, embedded images, scanned page — exercise SURGICAL/ALWAYS + table/formula/image paths against real pdf_oxide | M | End-to-end run only proved the models *load and run*; real recognition (layout → OCR/table/formula) on realistic pages is still unproven |
| 3 | **Raise coverage** 83% → ≥85%: converter guard/fallback lines (76%), remaining engine lines (82%) | M | Confidence in degradation paths |
| 4 | **Parity check** ported tests vs OKFgraph originals; document any behavioural drift | S | Keep the two codebases honest |
| 5 | **`slow` GPU job** in CI (onnxruntime CUDA) — optional | L | GPU provider path (`ort_providers`) untested |
| 6 | **PyPI publish** (0.1.0 or 0.2.0): `uv build`/twine, long description, classifiers | S | Distribution |
| 7 | **Consume from OKFgraph** (optional follow-up, per user constraint OKFgraph stays untouched for now): re-export shim or refactor of `cli.py`/`components/ingest.py` | M | Remove duplication, single source of truth |
| 8 | **Converter label fix**: add `display_formula`/`inline_formula` (pp_doc_layoutv3) to the formula-label tuple in `_full_structure_page_markdown` so layout-detected formulas route to the recognizer instead of OCR-as-text | S | Cheap, makes RapidLayout 1.2.1 formula detection usable |
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

- [ ] **116+** unit tests pass on a bare install (no optional deps)
- [ ] Integration suite green on a runner with `bobine[pdf-ingest]` + `bobine[formula]` installed (**demonstrated locally 2026-08-09**)
- [ ] Coverage ≥ 85% on `bobine/converter.py` + `bobine/pipeline.py`
- [ ] CI green on Python 3.10–3.13 (lint, format, unit, coverage)
- [ ] Test-PDF corpus committed under `tests/fixtures/` (or generated at test
      time with reportlab)
- [ ] RapidAI version pins (`versions.py`) match `pyproject.toml` **and** pass
      a real-install runtime smoke test
- [ ] `display_formula` / `inline_formula` routed to the formula recognizer
- [ ] Dual license metadata correct (`Apache-2.0 OR MIT`), vendored MIT notice
      intact, buildable wheel (`uv build` + `pip install` the wheel)
