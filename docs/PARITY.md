# Parity audit — bobine vs OKFgraph ingestion pipeline

**Date:** 2026-08-09 · **Method:** read-only; OKFgraph untouched.
**Result:** ✅ **35/35 pure-pipeline tests pass against bobine's modules**;
every code difference is intentional and catalogued below.

---

## 1. How the audit ran

- OKFgraph's `tests/test_ingest.py` was executed against **bobine's** modules
  (`bobine.assets/config/converter/engine/tables/versions`) via a `sys.modules`
  shim aliasing `okfgraph.ingest.*` → `bobine.*` (temp dir, outside both repos).
- Code drift was measured with `diff` per module, then each changed line was
  classified (intentional-with-reason vs accidental). No accidental drift found.

## 2. Test parity

| OKFgraph test file | In scope for bobine? | Result |
|---|---|---|
| `tests/test_ingest.py` — config, engine degradation, tables, assets, converter-init, versions | ✅ pure pipeline | **35/35 passed** via shim (subset below) |
| `tests/test_ingest.py` — `TestRouter`, `TestIngestPdfMethod` | ❌ graph/API layer (`okfgraph.router`, `ingest_pdf`) | out of scope by design (bobine exposes `ingest_document`; consuming from OKFgraph is roadmap #7) |
| `tests/test_converter.py` — GUI staging + router round-trip | ❌ GUI + `okfgraph.images` writer | out of scope (GUI is OKFgraph-only) |
| `tests/test_ingest_tool.py` — CLI clip/omni modes | ❌ OKFgraph CLI | out of scope |
| `tests/test_pdf_e2e.py` — router e2e, concept creation | ❌ router/concept layer | out of scope |

Shim-run subset (all passed): `TestConfig` (6), `TestEngineGracefulDegradation`
(5), `TestHtmlTables` (6), `TestAssets` (5), `TestHybridConverterInit` (3),
`TestVersions` (5), plus 5 more version/asset tests = **35**.

Every OKFgraph test with a bobine equivalent is mirrored in bobine's own
suite (`test_config`, `test_engine`, `test_tables`, `test_assets`,
`test_converter`, `test_versions`) — ported during Phase 1 and extended since.

## 3. Code drift catalogue (all intentional)

| Module | Changed / total | Drift | Reason |
|---|---|---|---|
| `assets.py` | 7 / 90 | typing modernization (`Tuple`→`tuple`, `"re.Match"`→`re.Match`, `.encode()` arg) | Python ≥3.10 cleanup; no behavior change |
| `tables.py` | 5 / 87 | same typing modernization | no behavior change |
| `config.py` | 14 / 94 | `min_formula_math_chars` 3 → 5; new `formula_layout_fallback` flag | Phase 8 (skip tiny formula fragments; opt-in layout fallback) |
| `versions.py` | 48 / 148 | env var `BOBINE_INGEST_ALLOW_UNPINNED` (legacy `OKFGRAPH_…` honoured); `_KNOWN_GOOD` pins → modern verified set; `rapid_latex_ocr` removed (vendored) | Phase 4 (OKFgraph's pins 1.5.2/1.0.13 **don't exist on PyPI**) |
| `engine.py` | 99 / 239 | RapidAI 3.x adapters: `RapidOCR()` no-kwargs, dataclass response normalization (layout/table/ocr), ndarray-safe box handling, `_np` guarded import, vendored `LatexOCR` import | Phase 5–6 (real-backend API drift, verified live) |
| `converter.py` | 286 / 829 | import rewrite (`okfgraph.ingest.*`→`bobine.*`); `_bbox_to_xyxy` bbox-format normalization; line-aware formula-box merge; text-layer-first tables/text in full-structure path; layout-equation fallback (`_layout_equation_boxes`); pixels→points conversion; `_MATH_LAYOUT_LABELS`; crop-None guards; render-page-duck-typing for pdf_oxide 0.3.x | Phases 1, 5–9 (each tracked in IMPLEMENTATION_PLAN) |

## 4. OKFgraph-side observations (for the future consumer work, roadmap #7)

- OKFgraph's `_KNOWN_GOOD` still pins `rapidocr==1.5.2` and
  `rapid_latex_ocr==1.0.13` — **neither exists on PyPI**; its
  `check_rapid_versions()` warns against the real install. When OKFgraph
  consumes bobine (or is updated), port the Phase 4 pin set.
- OKFgraph's tests exercise the graph layer (`OKFRouter`, `ingest_pdf`) that
  bobine deliberately excludes — the two suites are complementary, not
  overlapping; no behavioral contradiction detected in the shared surface.
- Behavioral defaults bobine changed (all surfaced by the drift catalogue):
  `min_formula_math_chars` 3→5; modern RapidAI pins; vendored formula OCR.

## 5. Conclusion

The extraction is **behaviorally faithful** on the shared surface (35/35
tests), and every deliberate change is documented with its reason. No
accidental drift. The two codebases stay in sync until OKFgraph consumes
bobine (roadmap #7), at which point the drift catalogue above becomes the
migration map.
