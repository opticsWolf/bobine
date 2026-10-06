# Changelog — bobine

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
