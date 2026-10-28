# Changelog — bobine

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
