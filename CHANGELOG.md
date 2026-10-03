# Changelog — bobine

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
