# bobine — Consolidated Implementation Plan

**Status:** v0.1.0 · dual-licensed (Apache-2.0 OR MIT) · `https://github.com/opticsWolf/bobine`
**Last updated:** 2026-08-08

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
| Version pinning / runtime check | ✅ extracted (`versions`) |
| Text-type document model + lint | ✅ new (`documents`, `markdown`) |
| Orchestration (single + batch) | ✅ new (`pipeline`) |
| Unit tests (fake pdf_oxide, no deps) | ✅ **99 passed**, 3 integration skipped |
| Coverage | ✅ **78%** (converter 76%, engine 43%, pipeline 87%) |
| Lint / format (ruff) | ✅ clean |
| CI (GitHub Actions) | ✅ core matrix 3.10–3.13 + integration job |
| Git / remote | ✅ `main` on GitHub, clean tree |
| Real-backend verification | ⬜ pending (needs `bobine[pdf-ingest]` install) |

---

## 3. Repository Layout

```
bobine/
├── pyproject.toml            # deps, extras, pytest/coverage/ruff config
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
│   └── pipeline.py           # convert_to_markdown / stage_images /
│                             #   ingest_document / convert_directory
└── tests/                    # 12 files, 99 unit + 3 integration tests
```

### Extras (all optional — the package imports with zero dependencies)

| Extra | Provides |
|---|---|
| *(core)* | `Pillow` |
| `[pdf-ingest]` | `pdf_oxide`, `rapidocr==1.5.2`, `rapid_latex_ocr==1.0.13`, `rapid_layout==0.2.0`, `rapid_table==1.0.3`, `numpy` |
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
6. **Dual license.** `Apache-2.0 OR MIT` (SPDX) — pointer `LICENSE` +
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

---

## 6. Open Work — Prioritized Roadmap

| # | Item | Effort | Why |
|---|---|---|---|
| 1 | **Verify against real backends**: `pip install -e ".[pdf-ingest]"`, run `pytest -m integration`; validate RapidAI version pins (`versions.py`) against the real installed stack | S | Biggest open risk: 3 integration tests never executed; pins inherited unvalidated |
| 2 | **Test-PDF corpus** (reportlab-generated at test time): digital text, math formulas, GFM tables, embedded images, scanned page — exercise SURGICAL/ALWAYS + table/formula/image paths against real pdf_oxide | M | Fake tests prove logic, not pdf_oxide output fidelity |
| 3 | **Raise coverage** 78% → ≥85%: converter guard/fallback lines, `engine.py` lazy-loaders (43% — only reachable with real RapidAI or tighter engine fakes) | M | Confidence in degradation paths |
| 4 | **Parity check** ported tests vs OKFgraph originals; document any behavioural drift | S | Keep the two codebases honest |
| 5 | **`slow` GPU job** in CI (onnxruntime CUDA) — optional | L | GPU provider path (`ort_providers`) untested |
| 6 | **PyPI publish** (0.1.0 or 0.2.0): twine/uv build, `README`/long_description, classifiers | S | Distribution |
| 7 | **Consume from OKFgraph** (optional follow-up, per user constraint OKFgraph stays untouched for now): re-export shim or refactor of `cli.py`/`components/ingest.py` | M | Remove duplication, single source of truth |

**Legend:** S = < 1 day · M = 2–3 days · L = 1+ week

---

## 7. How to Run

```bash
cd D:/User/Documents/Python/bobine
uv venv .venv && uv pip install --python .venv/Scripts/python.exe -e ".[dev,markdown]"

# unit suite (no native backends needed)
.venv/Scripts/python.exe -m pytest

# integration suite (requires bobine[pdf-ingest])
.venv/Scripts/python.exe -m pytest -m integration

# coverage + lint
.venv/Scripts/python.exe -m pytest --cov=bobine --cov-report=term-missing
.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .
```

---

## 8. Acceptance Criteria (definition of done for v1.0)

- [ ] 99+ unit tests pass on a bare install (no optional deps)
- [ ] Integration suite green on a runner with `bobine[pdf-ingest]` installed
- [ ] Coverage ≥ 85% on `bobine/converter.py` + `bobine/pipeline.py`
- [ ] CI green on Python 3.10–3.13 (lint, format, unit, coverage)
- [ ] Test-PDF corpus committed under `tests/fixtures/` (or generated at test
      time with reportlab)
- [ ] Version pins in `versions.py` validated against the pinned RapidAI
      versions in `pyproject.toml`
- [ ] Dual license metadata correct (`Apache-2.0 OR MIT`) and buildable wheel
