# Changelog — bobine

## Unreleased

- **Option C: version-scoped render-health gate (pdf_oxide < 0.3.79).** New
  `src/render_gate.rs` + `RenderHealthOpts` (`render_gate_enabled=true`,
  `render_gate_min_layer_words=30`, `render_gate_min_jaccard=0.35`,
  `render_gate_salvage_inline_math=false`, Python mirrors + getters): per
  page, before the expensive arms, OCR↔text-layer word-Jaccard below
  threshold on a substantial layer means corrupt render — discard the ONNX
  output, emit fast-path markdown, stay loud (`tracing::warn` per fire +
  `RenderGateStats` on the converter, appended to the P-D bench
  `report.md`). Suppression-first math (TexTeller spans on fired pages
  dropped) with opt-in inline `$…$` salvage. `build.rs` reads the resolved
  pdf_oxide version from `Cargo.lock` and only emits
  `pdf_oxide_pre_0_3_79` below the 0.3.79 cutoff — the dep bump
  deactivates the shim at compile time (registry fallback: assume old,
  stay safe). Unit-tested pure logic (tokens/Jaccard/decision/version
  scope) + no-model converter tests (disabled/thin-layer fail-open); all
  goldens unchanged. Filed upstream as #3098.

- **POTATR P2: option-A routing with claim/suppress + e2e golden.** Page-level
  pre-pass claims tables (`score>=0.8`, `>=2` words in rect, cell fill >=
  `potatr_min_text_fill`=0.2) and suppresses overlapping layout regions
  (IoU>0.5 plus caption/footer cover rule); each table emits anchored
  markdown, unclaimed regions use the existing arms (SLANet fallback
  intact). New `tests/fixtures/potatr_table.pdf` →
  `tests/golden/potatr_table_potatr.golden.md` (ignored
  `BOBINE_POTATR_PARITY=1` gate, full 20-cell GFM table) plus the
  Never-mode `tests/golden/potatr_table.golden.md`. Incidental fixes
  routing depended on: DETR `scale_factor` ones (3-4x box inflation),
  `flip_chars_topdown` (bottom-up chars vs top-down boxes), `figure_title`
  as caption, `<html>` wrapper strip + self-closing td/th, cover rule.
  Default stays `Slanet` — no behavior change unless opted in.
- **POTATR P1: postprocess port + parity gate.** `src/rapid_potatr.rs` ports
  `postprocess.py` (objects → structures → cells, rotations, NMS family),
  `metrics.cells_to_html` (thead/th + tbody/td, ET-faithful empty elements +
  escaping), and `inference` edge-linking/`tables_from_outputs`. Four
  reference unit tests ported green; ignored `tests/test_potatr_parity.rs`
  (`BOBINE_POTATR_PARITY=1`) reproduces the torch reference on chem-olive p5
  with byte-exact HTML (21 cells), caption, and grid — box drift 0.054px.
  No routing yet — behavior unchanged.
- **POTATR P0: model slot + preprocess (opt-in).** New `src/rapid_potatr.rs`
  (`POTATR_REPO`/`REV=e7f1ce67…` — `opticsWolf/potatr-onnx`, `download_potatr`,
  `Potatr::load`/`forward`, exact eval-preprocess replication with
  `PotatrGeometry`); `OnnxEngine` lazy `potatr` slot + `set_potatr_model` +
  CPU-pinned providers; `TableBackend::{Slanet,Potatr,Auto}` (default
  `Slanet`) + thresholds + `potatr_ort_providers` in config, Python
  bindings, and `tests/test_potatr.rs` preprocess parity vs the reference
  transform (max|d| 0.0175 ≤ 0.03). No routing yet — behavior unchanged.

## 0.8.0 — 2026-11-07

The 0.7.5 development line (TexTeller fp32 default, variant matrix)
was never tagged; it ships here together with the serving-stack
renewal (PP-OCRv5, PP-DocLayout_plus-L, first-party mirrors), then the
0.7.5 line:

- **fp16 parity gate.** New `tests/test_tex_teller_parity.rs` (ignored
  by default, `BOBINE_TEX_PARITY=1`): loads fp32 + fp16 from the unified
  repo and asserts raw-decode fp16==fp32 parity >= 27/30 over the labeled
  eval set. Measured 28/30 CPU, matching the documented matrix.
- **Bench harness precision tiers.** `tests/test_bench_ocr.rs` takes
  `BOBINE_BENCH_QUANT=fp32|int8|fp16` (outputs namespaced to
  `target/bench_ocr/<quant>/`); int8-vs-fp32 over the 196 ikat displays:
  195/196 identical outcomes, both tiers starved equally by the CID
  text-layer corruption (withheld per standing rule).
- **PP-OCRv5 server tier evaluated and rejected.** 88.1 + 84.5 MB
  (8x mobile), 3.1x slower (6.95 vs 2.23 s CPU on solitons p.0),
  quality at parity (40/178 differing line-slots, wins split both
  ways). Not mirrored; mobile pair stands.
- **Layout upgraded to PP-DocLayout_plus-L.** New first-party serving
  repo `opticsWolf/pp-doclayout-plus-l-onnx` (RT-DETR-L, 20 classes,
  Apache-2.0, byte-identical to the official PaddlePaddle export).
  `RapidLayout` detects the backend from the input signature (YOLO path
  kept for custom models; DETR path: direct 800x800 resize, /255,
  three inputs, `[class,score,x0,y0,x1,y1]` in original coords,
  threshold 0.5). The 20-class taxonomy maps onto the existing
  substring routing (`doc_title`/`paragraph_title`→headings,
  `formula`→math, `chart`/`seal`→figure crops); econ-tax p.0:
  13 overlapping YOLO boxes → 11 tight DETR regions, 477ms → 297ms
  CPU. Supersedes `opticsWolf/doclayout-yolo-onnx` (YOLO-lineage
  review closed by replacement).
- **OCR upgraded to PP-OCRv5.** New first-party serving repo
  `opticsWolf/rapidocr-ppocrv5-onnx` (det + rec + `ppocrv5_dict.txt`,
  Apache-2.0, byte-identical to the official PaddlePaddle exports).
  The rec export carries no charset metadata, so the dict travels with
  the weights (18,383 lines → 18,385 classes via blank+space
  alignment); `RapidOcr::load` takes an optional dict path, and the det
  output lookup is generation-robust (`sigmoid*` else first output).
  Formula fixture: v4 `RN/y` → v5 `x2/y2` with higher confidences.
  Supersedes `opticsWolf/rapidocr-onnx` (PP-OCRv4, still on the Hub).

- **TexTeller now defaults to fp32.** `ModelOpts::default()` and the Python
  bindings select `ModelQuantization::Fp32` on both CPU and CUDA. int8 is now
  the opt-in CPU speed tier (≈1.9× faster, occasional decode drift) and fp16
  the opt-in CUDA tier (≈1.2× faster, 29/30 exact parity). A warning now
  fires for fp16 without a CUDA provider, mirroring the existing int8+CUDA
  warning.
- **TexTeller fp16 decode fixed.** The fp16 merged decoder hoists its
  entry Casts outside the `If`, so ONNX Runtime requires every
  `past_key_values.*` input even on the prefill step. The KV loop used to
  omit them (fp32/int8 tolerate that) and failed with `Missing Input:
  past_key_values.0.decoder.key`. Prefill now feeds empty
  `[1,16,0,64]` caches — valid for all three merged exports and ignored
  by the no-past branch. fp16 reproduces the fp32 LaTeX on the reference
  fixture.
- **TexTeller variant benchmark.** `bench_texteller` now covers
  fp32 / fp16 / q8 / int8 (`--all`), can run the matrix in one process,
  and scores predictions against a labeled eval set (`--eval`). The set
  is built by the new `gen_texteller_eval` example from the ikat
  ground-truth displays rendered with `pdflatex` (30 formulas, exact GT).
  q8 stays benchmark-only: the serving surface still has no q8
  constructor, and the repo's parity gate continues to reject it.
- **Variant matrix measured** (30 labeled formulas, CPU + CUDA). fp16
  matches fp32 on 28/30 CPU and 29/30 CUDA, differing only in `\,`
  placement. int8 matches on 10/30 — mostly stylistic drift, but one
  formula collapses — and is 1.9× faster than fp32 on CPU, 1.8× slower on
  CUDA. q8 matches on 4/30 and degenerates into a 2318-character
  repetition loop on one formula. int8 output also varies by provider
  (15/30 identical across CPU and CUDA).

## 0.7.4 — 2026-10-31

- `embroider` 0.3.3 → 0.12.0 (nine minors, all new model slots;
  zero source changes — our six call sites are API-stable).
- Converter weights pinned: every live weight (TexTeller fp32/int8,
  RapidLayout, RapidOCR det/rec, SLANet-plus) carries a `*_REV` hub SHA
  next to its repo constant; probe and download both resolve the pin.
  Download arm delegated to `embroider::fetch_repo_file`.
- TexTeller fp32 + int8 → first-party unified serving repo
  `opticsWolf/texteller3-onnx` (both upstreams proven weight-identical;
  int8 byte-verified sha256 against the former Ji-Ha blobs). The unified
  repo also builds the `*_fp16` pair the upstreams lack, so
  `ModelPrecision::Fp16` filenames resolve (numerics gate open — the
  repo's own parity table has fp16 exact 3/3 on its sample; bobine-side
  measurement deferred to the next release, as is the int8-vs-fp32
  parity gate over the GT displays).
- Bench harness saves `runtime.json` (machine inventory) next to
  `report.md`/`mismatches.md`. New ignored guard test
  `unified_repo_fp32_loads` (download + session build).
- q8 deliberately not adopted: fails the repo's parity gate.

## 0.7.3 — 2026-10-29

- `office_oxide` 0.1.10 → 0.1.16 (read-path fidelity + resource-bound
  hardening upstream). One source-compat fix (`xlsx::Cell.formula` is a
  method now) plus new `Option` IR fields on struct literals. Office
  output strictly improved — blank-line structure, correct ordered-list
  numbering, footnote/caption text populated, deck bullets recovered,
  xlsx formulas emitted (goldens regenerated and reviewed hunk-by-hunk).
  Also fixed en route: `chem_olive_golden_stable` normalized CRLF
  checkouts (pre-existing autocrlf failure, blob is LF-only).

## 0.7.2 — 2026-10-29

- Full wheel matrix (packaging only, no code changes): CPython
  3.10–3.14 × {manylinux x86_64 + aarch64, windows x64, macOS arm64}
  (20 wheels + sdist). 0.7.1 shipped one wheel per platform, forcing
  sdist builds on 3.10/3.11/3.13. Deliberate skips, documented in
  `.github/workflows/release.yml`: macOS x86_64 (no Intel runners),
  musllinux (action runs container setup on the host — proven CI
  failure, sdist fallback stands), cp39 and older. `requires-python`
  floor raised 3.9 → 3.10 (3.9 EOL Oct 2025).

## 0.7.1 — 2026-10-28

- Deferred media trio (proposal_media_tables): `ImageMode::{Link,
  EmbedBase64}` + `max_embed_bytes` cap (default 256 KiB, MIME-sniffed
  data URIs; over-cap figures stay links) threaded through all three
  figure paths (ONNX branch, crop path, fast-path interleave + gallery
  with embed-aware dedup); `preserve_layout` fast-path mode (strict
  visual order, no heading/table merging); side-by-side caption fix
  (vertical-overlap gate was dead code — now co-extent + small-gap
  checked first, unit-pinned). Mirrored in Python
  (`ImageMode`, `image_mode`, `max_embed_bytes`, `preserve_layout`).
- P-D OCR benchmark harness: `tests/test_bench_ocr.rs` (`#[ignore]`-gated,
  Always mode, order alignment, normalized exact match, `report.md` +
  `mismatches.md`, rescore mode) — accuracy table WITHHELD: the ikat
  corpus (PLOS ONE CID fonts) renders glyph-corrupted under pdf_oxide
  0.3.78, so Always-mode OCR inherits the corruption (analysis +
  solitons counter-proof in `docs/benchmarks.md`). Also fixed en
  route: RapidOCR `rec_input` normalization now `(x/255-0.5)/0.5` per
  the PP-OCRv4 spec. GPU path validated (RTX 3090, ORT 1.28.1
  cuda13 + cuDNN 9.24).

## 0.7.0 — 2026-10-28

- JATS input adapter (proposal P-B): `HybridConverter::convert_jats` +
  `JatsDocument { md_path, title, n_formulas, n_figures, n_tables }` —
  publisher full-text XML to GFM via `mml.rs` (inline `$…$` fused with
  `merge_math`, display `$$tex \tag{N}$$`), real JATS tables, figures
  as `![alt](href)` (remote URLs preserved), sections as headings.
  No ONNX needed; `.xml`/`.jats` now route to it in `convert()`.
  Python: `bobine.convert_jats` + `converter.convert_jats` +
  `JatsDocument` + stubs.
- `mml.rs` (proposal P-A, 0.7.0-alpha): verbatim MathML→LaTeX port with
  692-fixture byte-parity gate vs the shimmed Python reference.

## 0.6.1 — 2026-10-28

- Post-OCR LaTeX sanitizer (proposal P-C): `sanitize_latex` repair pass
  over TexTeller output at all three OCR call sites (hybrid refinement,
  whole-region fallback, surgical splice) — command splitting (`\inL` →
  `\in L`, longest-known-prefix so `\infty}` is never shredded),
  unclosed-fence fixup (`\left` imbalance → `\right.`), same-level
  script separation (`x^a^b` → `x^a{}^b`), brace-balance hard-fail back
  to raw text. Gated by new `text.sanitize_ocr` (default true, mirrored
  in `ConverterConfig(..., sanitize_ocr=...)` + stubs). New `sanitize`
  module; `mml.rs` (P-A) will reuse `split_cmds` directly.

## 0.6.0 — 2026-10-06

- Hub-cache alignment (cache-alignment plan Phase 2): converter models
  fetch through the standard HuggingFace hub cache in cache-mode — no
  more `local_dir` side-copies into `<output>/.cache`. `cache_dir`
  parameters now mean the **hub-cache root** (override); `ingest_document`
  takes an optional `cache_dir` (`None` = env-resolved standard cache),
  so okfgraph's default temp-dir ingest downloads once per machine
  instead of per document. The first fetch after upgrading re-downloads
  once (the old flat layout is not reused); repeats never re-download
  (blob reuse after a cheap HEAD).
- `OnnxEngine::model_status()` — one `embroider::CacheReport` per family
  (TexTeller variant, layout, OCR det+rec, SLANet-plus) via embroider's
  shared `cache_info_files` (floor now `embroider>=0.3.3,<0.4`);
  `ensure_models` logs the per-family cached/missing preflight. Python:
  `bobine.model_status(cache_dir=None)` with the same dict keys as
  okfgraph's `model_info`.
- Removed `rapid_table::slanet_plus_dest` (the flat destination no longer
  exists) and the `texteller_int8/` subdir (the int8 repo's own hub
  folder isolates variants). Fixed a stale doc: the default formula
  quantization is `Int8` (was documented as `Fp32`).

## 0.5.13 — 2026-10-03

- ONNX plumbing on `embroider >=0.3, <0.4` (same floor as okfgraph;
  policy API unchanged, full suite green against 0.3.1).
- Docs: implementation plans removed (superseded), README restructured
  with Performance overview, embroider ownership boundary documented.

## 0.5.12

- ONNX plumbing on `embroider 0.2` (`SessionPolicy`, `apply_providers`,
  EP-availability CUDA probe; plan-onnx-only Phase 2).
- ONNX Runtime extras pinned to `==1.29.0`, the single binary shared with
  okfgraph (`cpu` XOR `gpu`; plan-onnx-only Phase 3).
- Engine startup logs the shared-runtime report (resolved dylib, CUDA
  usability) via `embroider::report()`.
- Vision slots run the untuned `ort_defaults()` policy — pinned by the
  `vision_slots_never_use_text_policy` test; shared `parse_owner_name`
  validation for HF fetches (plan-onnx-only Phase 4).
- No conversion changes: golden outputs byte-identical to 0.5.11.
