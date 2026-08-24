# bobine_rs — Consolidated Status, Gaps & Implementation Plan

> Rust-first rewrite of bobine. Template: `legacy/docs/IMPLEMENTATION_PLAN.md`
> (pure-Python v0.2.0). This file is the single source of truth for what is
> done, what is missing, and in which order it will be built.
>
> Last updated: rust_dev @ `2eaadf8` (docs: architecture + quickref).

---

## 1. Goal

Parity with legacy Python bobine's **user-facing surface** — one-liner
ingestion (`ingest_document`, `convert_directory`), staged asset store,
document model, lint hook — on a pure-Rust core with PyO3 bindings, no
Python ML dependencies, distributed as `bobine` on PyPI.

## 2. Status Summary

| Layer | State |
|---|---|
| PDF fast path (pdf_oxide → markdown) | ✅ done |
| Routing NEVER / AUTO / SURGICAL / ALWAYS | ✅ done |
| Formula box detection (font+unicode heuristics, line-aware merge) | ✅ done |
| TexTeller ONNX formula OCR (hf-hub download, ort inference) | ✅ done |
| RapidLayout (DocLayout-YOLO) + full-structure page pipeline | ✅ done |
| RapidOCR (DBNet det + CRNN rec, CTC) | ✅ minimal port done |
| Office conversion (office_oxide auto-detect) | ✅ done |
| HTML → GFM tables | ✅ done |
| PyO3 bindings (`import bobine`, maturin, .pyi stubs) | ✅ done |
| Docs (architecture, quickref, this file) | ✅ done |
| Tests: 81 unit + 9 integration, 0 failures | ✅ done |
| Accuracy parity Int8-CPU vs Fp32-CUDA (10-formula corpus) | ✅ measured — 9/10 correct each, misses on different examples, rest byte-identical after normalization |
| Pipeline layer (assets/documents/pipeline modules) | ✅ Phase 2 done (`82d36bd`) |
| Markdown linting | ⚠️ partial — callback hook only, no native rules (see Known Gaps) |
| Scanned-table recognition (RapidTable) | ✅ Phase 3 done (slanet-plus, auto-download) |
| DB unclip det postprocess + rotated crops | ✅ Phase 3 done |
| ocr_lang config knob | ✅ Phase 3 done (charset fallback) |
| CI on branch + release workflow | ❌ Phase 4 (CI triggers fixed during merge prep) |
| Coverage fakes (converter logic without real PDFs) | ✅ Phase 5 done (`PdfSource` seam; converter fns 22% → 77%) |

## 3. Repository Layout

```
├── src/                Rust core (10 modules, ~2 300 LOC)
├── python/bobine/      PyO3 shim + type stubs        (import bobine)
├── tests/              cargo integration tests + CC BY 4.0 fixtures
├── docs/               architecture.md · quickref.md · IMPLEMENTATION_PLAN.md
├── Cargo.toml          workspace (cdylib + rlib)
├── pyproject.toml      maturin; dist name = bobine 0.3.0
└── legacy/             frozen pure-Python bobine v0.2.0 (reference)
```

## 4. Open Work — Prioritized Roadmap

Legend: **S** < 1 day · **M** 2–3 days · **L** 1+ week

### Phase 1 — Quick wins & wiring (S)

| # | Item | Effort | Why |
|---|---|---|---|
| 1.1 | **Wire `wrap_code_blocks` into the fast path** — currently dead code; call from `route_page` when `detect_code_blocks`, splice fenced blocks like Python does | S | Config knob exists but does nothing |
| 1.2 | **CI triggers on `rust_dev`**: add branch to `.github/workflows/CI.yml` push list; add `ORT_DYLIB_PATH` setup step (download onnxruntime release asset or pip install into venv and export path); skip-or-env for PDF tests | S | Zero CI runs today — regressions ship silently |
| 1.3 | **Expose `ort_providers` through to `ort::Session` builder** (`config.ort_providers` field already exists but is ignored by every session constructor); add `device` config sugar mapping cuda→CUDAExecutionProvider | S | GPU path untested otherwise (legacy roadmap #5) |

### Phase 2 — Pipeline layer (M) — restores legacy one-liner UX

| # | Item | Effort | Why |
|---|---|---|---|
| 2.1 | **`assets.rs`** — `stage_images_as_okf_assets`: SHA-256 content-hash ids (`img_<16hex>`), copy bytes into `_assets/`, rewrite `![](local)` → `![](okf-asset://<id>)`, skip http/data/already-staged links. Port of `assets.py` (~100 LOC Python → ~150 Rust, `sha2` + `regex`) | M | Core of the ingest contract |
| 2.2 | **`documents.rs`** — `Document{id,title,description,body,type,tags,metadata}`, frontmatter parse/dump (minimal YAML subset or `serde_yaml`), `load_markdown_document()`, `wrap_thoughts()` | M | Legacy documents.py parity |
| 2.3 | **`pipeline.rs`** — `ingest_document(path, output_dir, config, lint)` → `ConvertedDocument`; `convert_directory()` batch; expose both via PyO3 so Python gets `bobine.ingest_document(...)` one-liners back | M | The main missing UX |
| 2.4 | **Progress & cancellation callbacks** in PyO3: pass Python callables for `on_page(idx,total)` / `should_continue()`; check between pages inside `convert_pdf` | S | Legacy convert_to_markdown signature parity; long conversions are currently uncancellable/silent |
| 2.5 | **`markdown.rs` lint hook** — optional: accept any Python object with `.lint(content)/fix(content)` (mordant) behind a callback instead of a Rust linter re-write | S | Keeps Rust core lean; mordant stays Python-side |

### Phase 3 — Scanned tables & OCR quality (M–L)

| # | Item | Effort | Why |
|---|---|---|---|
| 3.1 | **RapidTable (slanet-plus)** — table structure ONNX: load model slot #4 in engine, feed crop + OCR lines, emit HTML → existing GFM converter. Replaces `[table: label]` placeholder on scans | L | Last missing recognizer of the four-model stack |
| 3.2 | **OCR det post-processing upgrade** — replace flood-fill bounding boxes with proper DB unclip (polygon offsetting) or at minimum min-area-rect rotation handling; improves multi-column scan reading order | M | Current contour boxes are axis-aligned only; rotated text degrades |
| 3.3 | **`ocr_lang` config** — plumb language-specific rec model selection | S | Legacy parity (currently en-only charset default) |

### Phase 4 — Distribution (S)

| # | Item | Effort | Why |
|---|---|---|---|
| 4.1 | **Release workflow**: tag `v*` → maturin build matrix (3 OS × py3.10–3.13), wheel smoke test, trusted publish to PyPI as `bobine`. Reuse lessons from legacy release.yml (merge-multiple artifact corruption fix applies verbatim) | S | Legacy roadmap #6 carried over |
| 4.2 | **PyPI trusted publisher** (user-side): Project `bobine` · Workflow `release.yml` · Environment `pypi` | S | Blocks 4.1 final step |
| 4.3 | **Merge `rust_dev` → `main`** once Phases 1–2 land; legacy/ stays frozen in-tree | S | Single-branch simplicity going forward |

### Phase 5 — Quality (ongoing)

| # | Item | Effort | Why |
|---|---|---|---|
| 5.1 | ~~Fake-pdf_oxide unit layer~~ ✅ **done** (`6bc74d8`) — `src/pdf_source.rs` trait + in-memory fakes; converter.rs 77 % fn / 71 % line coverage (cargo-llvm-cov) | L | Met the ≥70 % target |
| 5.2 | ~~Corpus regression assertions~~ ✅ **done** — `tests/test_golden.rs`: deterministic fast-path conversion of all fixtures vs checked-in goldens (`tests/golden/`); regenerate with `BOBINE_UPDATE_GOLDENS=1` | M | Catches silent quality drift |
| 5.3 | **GPU CI job** (optional) — onnxruntime-gpu runner, exercises CUDA provider path | L | Legacy roadmap #5 |

### Deferred (by design — not gaps)

### Known Gaps

| Gap | State | Closure path |
|---|---|---|
| **Native markdown lint** — legacy ran mordant's linter (fix MD009/MD012/MD047; detect MD001/MD031/MD033). The Rust core currently ships only the `LintFn` callback hook (`pipeline.rs`); consumers must bring their own linter (e.g. a Python callable wrapping mordant-py) or skip linting entirely. | Partial — hook exists since `82d36bd`, zero native rule implementations | Blocked on **mordant landing on crates.io**: the real linter lives in mordant-py's `src/linter.rs` (2,205 lines, AST-based via rushdown types), currently unpublished and unpublishable as-is. Once a `mordant`/`rushdown` lib crate is published, add an optional bobine feature that depends on it and swap `lint_markdown()` internals behind the existing signature. A minimal self-contained port of the six rules remains possible as a stopgap if the crate dependency is unacceptable long-term. |

### Deferred (by design — not gaps)

| Item | Reason |
|---|---|
| Quantized formula weights | ✅ Int8 default since v0.3.9; **upgraded to KV-cached Ji-Ha/TexTeller3-ONNX-dynamic export in v0.4.2**: 319 MB total, ~10 ms/step vs ~26 ms fp32 -> formulas 0.44-0.64 s (2.4x). FP16 still unavailable (converters break on merged If-graph) |
| FP8 | Does not exist for TexTeller anywhere (both HF repos checked); ORT CPU EP has no fp8 kernels — not applicable |
| KV-cache decoder (`decoder_with_past_model.onnx`) | optimum KV-state divergence unresolved; merged-decoder greedy is correct |
| OKFgraph consumption shim (legacy roadmap #7) | Blocked by user constraint — OKFgraph untouched |
| Formula accuracy alternative (legacy roadmap #9) | Done — TexTeller *is* the upgrade |
| Vendored RapidLaTeXOCR | Removed by design; TexTeller replaces it |
| Int8 quantization of the Rapid models (layout/det/rec/table) — **measured & rejected** (2025 session, uncommitted experiment): ORT static QDQ int8 (u8 act / s8 per-channel weights, MinMax calibration on synthetic pages) vs fp32 on Ryzen 5950X CPU. Layout 1024²: **484 -> 660 ms (37% slower)**, output cosine 0.79; det 960×736: 66 -> 87 ms, cosine 0.90; rec: 36 -> 41 ms and decoded strings strictly worse; table: 13 -> 27 ms (2x slower). Sizes: only layout shrinks meaningfully (75 -> 20 MB); the small models gain nothing (QDQ overhead ~ their weight footprint). QOperator format fails outright on all graphs (`AttributeError NoneType.data_type` in the quantizer - paddle2onnx node patterns unsupported). Contrast with TexTeller Int8 (2.4x win): that decoder is a bandwidth-bound transformer where big MatMuls dominate; these are compute-bound conv nets where MLAS fp32 kernels already win and QDQ dequant churn costs more than u8s8 saves. Do not retry without a different runtime (e.g. OpenVINO EP) or fused QOperator exports from upstream |
| **Rapid models fp32 on CUDA vs CPU — measured** (RTX 3090 vs Ryzen 5950X, `examples/bench_rapid.rs`): layout 1024² **457 -> 37 ms (12.3x)**, OCR det+rec 1024² **146 -> 40 ms (3.6x)**, table 700×400 32 -> 57 ms (**1.75x slower**) - CONFIRMED by 30-rep rerun: 700×400 CPU 30.6ms vs GPU 64.1ms (2.1x); 1024² CPU 137ms vs GPU 1170ms (**8.6x - catastrophic**: SLANet's HardSwish/Cast-heavy graph fragments across devices, PCIe round-trips scale with input size; ORT warns of unassigned nodes). Guidance: offload layout+OCR to the GPU when `ort_providers` includes cuda; keep table recognition on CPU. NOTE: current config applies one `ort_providers` list to all sessions, so enabling CUDA today also slows table recognition - per-model provider control is the follow-up if scanned-document GPU throughput matters |
| **pdf_oxide geometric layout vs RapidLayout — measured** (v0.4.6, 4-page born-digital math paper, CPU): NEVER fast path **57 ms** vs ALWAYS full structure **261 s** (~4600x) - pdf_oxide wins overwhelmingly on text-layer PDFs, which is exactly why routing is text-layer-first. RapidLayout earns its cost only where geometry is blind: scans, figures, visual formulas (12 $$ blocks detected vs 0). Two v0.4.5 defects found by this comparison and fixed in v0.4.6: (1) `read_labels` treated a missing/empty metadata key as Some(empty vec), defeating the DocStructBench fallback -> every region unlabeled -> no headings, no formula/table routing; now also parses Ultralytics `names` metadata. (2) Known gap confirmed: overlapping layout regions each emit their text-layer content (page-2 phrase repeated 3x, ~4x word inflation) - region overlap dedup needed before ALWAYS mode is production-quality on born-digital input |

## 5. Acceptance Criteria (definition of done for v0.3.0)

- [ ] All 45 current tests green; new pipeline-layer unit tests green
- [ ] `bobine.ingest_document("paper.pdf", "out/")` works end-to-end from Python:
      markdown on disk, `_assets/` store populated, links rewritten to `okf-asset://`
- [ ] `bobine.convert_directory("docs/", "out/")` batches without leaking handles
- [ ] Progress + cancellation callbacks functional (verified by test)
- [ ] CI green on `rust_dev` (then `main`): fmt + clippy + test matrix with ORT set up
- [ ] Release workflow builds wheels for 3 OS × py3.10–3.13; `bobine==0.3.0` publishes
- [x] Converter function coverage ≥ 70 % (fake layer): 77 % functions / 71 % lines
- [ ] Docs updated same-commit with any API change

## 6. How to Run

```bash
# dev install (Rust toolchain + maturin required)
maturin develop --release

export ORT_DYLIB_PATH=<venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll

cargo test                     # 45 tests
python -c "import bobine; print(bobine.__doc__)"

# rebuild graph index after refactors
codegraph init   # idempotent; auto-sync watches files
```| FP16 model generation & benchmarking | Decoder: no compatible export exists (onnx-community lacks KV-cache; converters break on merged If-graph). Encoder: MEASURED and REJECTED, both sources — Ji-Ha true-f16 export (CPU 353 ms vs int8 176 ms; CUDA 90 ms vs fp32 38 ms) AND self-downcast of the fp32 graph via onnxconverter-common keep_io_types (CPU 268 ms vs 224 ms; CUDA 54.6 ms vs 36.3 ms). ORT's CUDA EP gains nothing from f16 weights alone: without fused fp16 attention kernels the inserted Cast nodes only fragment execution.
