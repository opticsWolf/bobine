# bobine — Code review (2026-09-09)

- **Commit:** `ca1e941` — v0.4.9: per-slot provider control, auto-GPU layout/OCR, CPU-pinned tables
- **Tree state:** dirty (~2183 insertions uncommitted: `src/converter.rs` +1588, `src/tables.rs` +297, `src/pdf_source.rs`, `src/config.rs`, goldens). Line numbers below refer to the dirty tree.
- **Method:** CodeRadar `explore` + `get_smells` (651 findings) + `dead_code` (100 findings @ ≥0.60) + `find_clones` + `find_scaffolding` (73 findings), plus targeted reads of `src/converter.rs` (2883 lines, 92 fns), `src/engine.rs` (541 lines), `src/config.rs` (251 lines), `src/tex_teller.rs` (612 lines).
- **Scope:** `src/`, `tests/`, `examples/`, `python/bobine/__init__.pyi`, `legacy/` (reference only — not a refactor target).

## Verdict

Architecture is sound: per-model modules (`tex_teller`, `rapid_layout`, `rapid_ocr`, `rapid_table`), lazy `OnnxEngine`, `PdfSource` seam (Phase 5), golden tests, disciplined provider fallback. Duplication is low (1 clone group), there is zero `TODO/FIXME` debt in `src`, and test fakes are correctly gated behind `#[cfg(test)]`.

The single dominant risk is size/complexity concentration in `src/converter.rs`, above all one god-function (`full_structure_page_markdown`: 639 LOC, cyclomatic 88, nesting 13). Everything else is P1 and below.

---

## P0 — must fix

### 1. `full_structure_page_markdown` is a god-function — `src/converter.rs:1566`

CodeRadar: `long-method` (LOC=639), `high-cyclomatic-complexity` (88), `deep-nesting` (13). Confirmed by reading: one function does render → text-layer math-box extraction → layout → reading-order sort → figure dedup → glyph-ownership assignment (`O(chars × regions)`) → line clustering → majority-vote assignment → per-region dispatch (table / math / figure / text / caption) → image staging → markdown assembly.

Consequences:

- Untestable as a unit (the glyph-ownership and dispatch logic can only be exercised end-to-end).
- The hottest loop (glyph ownership over every char × every region) has no isolated perf hook.
- The per-label routing branches duplicate scale/crop/fallback handling.

Recommended split (preserve the existing glyph-vs-strip comments — they are excellent — on the extracted functions):

- `assign_glyph_owners(chars, regions, scale) -> (owners, glyph_owner)`
- `cluster_owned_lines(chars, owned, glyph_owner, rank) -> region_lines`
- `render_region_table / _math / _figure / _text (...) -> Option<String>`
- `dispatch_sorted_regions(...) -> Vec<String>` (owns the `claim_order` + `rank` logic)

The `CLAIM_*` priority scheme (`region_claim_priority`, `src/converter.rs:~80`) is good design — keep the ordering, just move each arm's body out.

---

## P1 — should fix soon

### 2. `RapidTable::recognize` — `src/rapid_table.rs:261` (~150 LOC, cyclomatic 21)

Argmax loop + bbox-decode + canvas→original rescale + OCR furniture filter + `match_cells` + `build_html` in one body. Specific concern at ~line 318:

```rust
let ti = t.min(bbox_steps - 1); // bbox head may emit fewer steps
```

When the structure and bbox heads disagree in length this silently reuses the last bbox, potentially misaligning trailing cells. Log (at least `tracing::debug!`) when the clamp engages, and consider truncating the structure loop to `bbox_steps` explicitly.

### 3. `TexTeller::autoregressive_decode_kv` — `src/tex_teller.rs:418` (122 LOC, cyclomatic 10, nesting 5)

Logic is correct (prefill + pinned encoder cache + rolling decoder cache) but fragile: 48 stringly-typed IO names (`present.{i}.decoder.{k}` / `past_key_values.{i}.*`). Any model re-export breaks this at runtime rather than at load. Recommendations:

- Assert the expected IO schema at `TexTeller::load*` time (fail fast when `kv_cache` mismatches the graph).
- Extract `prefill_step()` / `cached_step()` to halve the nesting.
- Keep the "true branch emits broken encoder cache" comment — it is load-bearing knowledge.

### 4. Supporting hotspots (all confirmed in source)

| Symbol | Location | Signals |
|---|---|---|
| `math_density` | `src/converter.rs` | 108 LOC / cyclomatic 16 / nesting 4 |
| `nearby_caption_text` | `src/converter.rs` | 60 LOC / cyclomatic 11 / nesting 9 |
| `region_claim_priority` | `src/converter.rs:~80` | nesting 8 (flatten with early returns; keep ordering) |
| `rect_subtract` | `src/converter.rs` | 51 LOC / cyclomatic 10 |
| `stage_images_as_okf_assets` | `src/assets.rs` | 64 LOC / cyclomatic 10 |
| `surgical_page_markdown` | `src/converter.rs:323` | 54 LOC / cyclomatic 11 |
| `route_page_inner` | `src/converter.rs:250` | 50 LOC / cyclomatic 10 / nesting 4 |
| `apply_providers` | `src/engine.rs:~20` | 47 LOC / cyclomatic 13 |
| `examples/*/main` (×5) | `examples/` | 57–104 LOC, nesting up to 7 — acceptable for examples; do not copy into `src` |

### 5. `recognize_formula_capped` is not panic-safe — `src/engine.rs:335`

```rust
tt.max_tokens = max_tokens.clamp(16, 1024);
let latex = tt.recognize(image_path)?; // on Err, the restore below never runs
tt.max_tokens = crate::tex_teller::MAX_TOKENS;
```

Any `recognize` error leaves the converter with a clamped decode budget for all subsequent pages. Fix with a guard:

```rust
let prev = std::mem::replace(&mut tt.max_tokens, max_tokens.clamp(16, 1024));
let r = tt.recognize(image_path);
tt.max_tokens = prev;
let latex = r?;
```

### 6. `OnnxEngine::convert_pdf` has dead branches — `src/engine.rs:~455`

The `Never | Auto` and `Surgical | Always` match arms are byte-identical (`to_markdown` with fallback to `extract_text`). Real routing lives in `converter.rs:route_page_inner`; this match either never got wired or is a port stub. Delete the match (single path) or delegate to `HybridConverter`.

---

## P2 — cleanup

### 7. Error swallowing hides ONNX failures

- **Surgical-scanned path is silent, AUTO is not.** `route_page_inner` (`src/converter.rs:250`) logs `warn!` when the full-structure pass errors under AUTO, but the Surgical-scanned branch uses `if let Ok(Some(md))` with no warning. Make them consistent.
- **`Ok(None)` conflates "render failed" with "no regions".** `full_structure_page_markdown` (`src/converter.rs:1566`) returns `Ok(None)` when `render_page_image` fails — callers cannot distinguish infrastructure failure from an empty page. Return the error or `warn!` at the site.
- **Production `unwrap()`s on untrusted input:**
  - `src/converter.rs:925` — double `unwrap()` on `asset.bbox_pts`
  - `src/converter.rs:530` — `chars().next().unwrap()` on potentially empty input
  - `src/converter.rs:1699` — `glyph_owner[i].unwrap()` inside the tally loop (invariant-held, but a `debug_assert!` + `let-else continue` is cheaper than a panic on a corrupt page)
  - Replace with `let-else` + fallback. (All other `unwrap/expect` hits in `src` are under `#[cfg(test)]` or `expect("static regex")` — fine.)

### 8. `ConverterConfig` bloat — `src/config.rs:~90–200`

30+ fields; CodeRadar flags 6-parameter lists on `convert_to_markdown`, `ingest_document`, `convert_directory`, `_region_text`, `stage_images_as_okf_assets`. The v0.4.9 per-slot provider overrides (`encoder / decoder / layout / ocr / table_ort_providers`) are the right feature but tip the struct over. Suggested grouping (keep wire compat with `#[serde(flatten)]`):

- `RenderOpts { render_dpi, formula_dpi, formula_pad_pts, ... }`
- `ModelOpts { precision, quantization, formula_backend, ... }`
- `ProviderOpts { base: ort_providers, encoder, decoder, layout, ocr, table }`

### 9. Minor

- Duplicated `// Constants` header block, `src/converter.rs:34–38`.
- `HybridConverter.config` carries `#[allow(dead_code)]` (`src/converter.rs:~100`) while `OnnxEngine` holds a clone. Two sources of truth risk silent divergence if one side ever mutates. Prefer `Deref<Target = ConverterConfig>` into the engine's copy or a shared `Arc<ConverterConfig>`.
- `cuda_available()` (`src/engine.rs`) probes with a real `Session::builder()` cached forever in a `OnceLock`. Correct as a fast path, but a transient failure poisons the process for its lifetime — document that, or re-probe on explicit provider-override failure.

---

## Explicit non-issues (CodeRadar noise — do not act without verification)

- **Dead-code (100+ hits, ≥0.60 confidence) is ~90% false positives.** The detector does not treat `cdylib`/`rlib` public API, PyO3 (`src/py_bindings.rs`), or `legacy/` Python entry points as roots, so it flags live code such as `HybridConverter::convert`, `OnnxEngine::new`/`load*`, `TexTeller::recognize`, all `PdfSource` impls, and every test as "unreachable". The `transitively-dead` entries (`full_structure_page_markdown`, `surgical_page_markdown`, `route_page`, `route_page_inner`) are provably live via `route_page_inner` call edges. Verify with `affected()` before removing anything from this list.
- **Clones: exactly 1 group** (`tests/test_golden.rs::corpus_matches_goldens` vs `figure_corpus_matches_goldens`, type-3, similarity 0.86). Real but small — extract a `run_corpus_golden(kind)` helper. Otherwise duplication is impressively low for a Python-port codebase.
- **Scaffolding (73 hits): benign.** 27× `...` placeholder bodies are all in `python/bobine/__init__.pyi` (expected for stubs); 46× "Phase N" markers live in `docs/` + `IMPLEMENTATION_PLAN.md` (plan tracking, not code debt). No `TODO`/`FIXME`/`HACK` in `src`.
- **Test doubles:** `FakePdf`/`FakePage` are correctly behind `#[cfg(test)]` (`src/pdf_source.rs:259`). They do not bloat release builds.

---

## Suggested order

1. P0 split of `full_structure_page_markdown` (biggest testability + perf-visibility win).
2. P1-5 (`max_tokens` guard) + P1-6 (`convert_pdf` dead arms) — both under 10 lines.
3. P1-2/P1-3 extractions + bbox-clamp logging.
4. P2-7 logging consistency, then P2-8 config grouping (serde-compat care needed).
5. Clone helper in `tests/test_golden.rs`.
