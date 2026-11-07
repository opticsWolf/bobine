# Proposal: POTATR page-level table arm

Status: proposed (2026-11-07). Export verified (`potatr-export/`: `potatr_dyn.onnx`,
119 MB, torch-vs-ORT max|d| ≤ 6e-5, ~340 ms CPU/page); ikat GriTS gate passed
(micro GriTS-Con F1 **0.900**, TEDS 0.873 over 39 GT tables); mirror live
(`opticsWolf/potatr-onnx` rev `e7f1ce67…`, sha256-verified). No bobine code
points at it yet.
Scope: bobine Rust core (new `src/rapid_potatr.rs`, `engine.rs`, `config.rs`,
`converter.rs`, Python bindings) + tests + docs
Related: [architecture.md](architecture.md) (§9 table-slot review),
[benchmarks.md](benchmarks.md) (rapid-pipeline table), `potatr-export/ASSESSMENT.md`
(export + gate record, untracked local), thoughts `bobine-analysis` (mirror + gate)

---

## 1. Goal and non-goals

**Goal:** a page-level table arm for pages that have words. One POTATR inference
per qualifying page replaces the whole current table chain on that page —
layout `table` regions, `table_block` Stages 1–3 (`converter.rs:386`), and the
OCR-crop → SLANet-plus → `match_cells` → `build_html` path (`rapid_table.rs:256`,
`rapid_table.rs:77`, `rapid_table.rs:104`) — with direct text-bearing HTML plus
table↔caption/footer edges the pipeline does not produce today.

**Non-goals (v1):**
- No deletion of SLANet-plus or `table_block`. The plan is additive with
  suppression; every phase is revertible by config (`table_backend`).
- No crop-level use inside `table_block` (destroys page context, multiplies
  cost per page — see §2).
- No wordless pages (structure-without-text ships empty cells; GriTS-Con ≈ 0).
- No running text / figures / formulas / reading order changes; layout plus
  the other region arms stay exactly where they are.
- No multi-page tables / cross-page merging (paper lists it as future work).

## 2. The two contracts (why a new arm, not a swap)

| | SLANet-plus (incumbent) | POTATR (proposed) |
|---|---|---|
| Input | single **table crop** pixels (+ OCR lines for text) | **full page** pixels (longest side → 1000); words downstream only |
| Output | structure tokens + cell boxes (no text, no detection) | table/row/col/header/spanning/caption/footer boxes + 250×250 edge matrix + cells + HTML |
| Text | matched from OCR lines by geometry (`match_cells`) | assigned from page words by geometry (post-network) |
| Detection | none (the region *is* the detection, supplied by layout) | full page-level detection with scores |
| Provenance | `opticsWolf/slanet-plus-onnx` @ `40bbe318…` | `opticsWolf/potatr-onnx` @ `e7f1ce67…` (MIT end to end) |
| Measured | never had a GT content gate on ikat | ikat micro GriTS-Con 0.900, TEDS 0.873 |

The `table_block` fallback chain (structured extract → line dump → OCR+structure)
exists *because* SLANet produces structure without text. POTATR produces
text-bearing HTML directly, so plugging it into Stage 3 would leave Stages 1–2
running pointlessly and strand its caption/footer/edge outputs with nowhere to go.

Key export facts the implementation depends on (all verified in `potatr-export/`):
- `POTATR.forward` takes images only (`[B,3,H,W]` + padding mask); words never
  enter the network (`model.py:114-128`).
- ONNX signature: input `image` `[1,3,H,W]` float32, **dynamic H/W axes**;
  outputs `pred_node_logits` `[1,250,17]`, `pred_boxes` `[1,250,4]` (normalized
  cx,cy,h,w), `pred_edge_logits` `[1,250,250,2]`.
- Eval preprocess (must be replicated exactly, §4): PIL `image.resize` longest
  side → 1000 with **default resample** + ImageNet mean/std normalize
  (`datasets.py: get_transform("eval")`, `RandomMaxResize`).
- Thresholds from reference code: node score 0.8 (`evaluate.py`
  `compute_predicted_edges`), edge 0.5 (`inference.py` default).
- Tokens format: `[{bbox, text, block_num, line_num, span_num}]` (block/line/span
  order keys required by `extract_text_from_spans`); PyMuPDF `get_text("words")`
  maps 1:1.

## 3. Target architecture

Page-scope pre-pass in the converter, above the region loop:

```
page render + page words (text layer, or page-level OCR words for scans)
  → words < MIN_WORDS? → current pipeline unchanged (zero behavior change)
  → POTATR forward (≈340 ms CPU, once per page)
      → claimed table boxes + edge-linked caption/footer boxes
      → HTML blocks emitted at visual position in reading order
      → suppress layout regions overlapping claims (IoU rule, §6)
      → unsuppressed regions → existing arms (table_block incl. SLANet fallback)
```

Claim bookkeeping rides the existing `RegionCtx` bundle (`converter.rs:197`;
the P0 comment there already anticipates page-level inputs) plus the `blocks`
push path the line-claim loop uses (`converter.rs:850` area). Suppression reuses
the `region_claim_priority` / `CLAIM_NONE` mechanism (`converter.rs:83-108`).

## 4. Phase P0 — model slot + preprocess parity (small)

**New file `src/rapid_potatr.rs`** (naming follows `rapid_ocr.rs`,
`rapid_layout.rs`, `rapid_table.rs`):

- Constants mirroring `rapid_table.rs:173-180`:
  `POTATR_REPO: (&str,&str) = ("opticsWolf","potatr-onnx")`,
  `POTATR_FILENAME = "potatr.onnx"`,
  `POTATR_REV = "e7f1ce67372e5f3972d7c93c8dc639258c6f1944"`,
  plus `POTATR_LONGEST_SIDE: u32 = 1000`.
- `download_potatr(cache_dir)` via `hub_cache::hub_fetch` (same shape as
  `download_slanet_plus`, `rapid_table.rs:185-199`).
- `Potatr::load(path, providers)` — ort session, dynamic axes (no fixed canvas).
- `preprocess(img: &DynamicImage) -> Array4<f32>` — **exact** replication:
  1. read `RandomMaxResize` + the inner `ToTensor`/`Normalize` in
     `potatr/datasets.py` and pin the constants in comments (resample mode,
     mean, std — verify, do not assume);
  2. longest-side → 1000 with the `image` crate filter matching PIL default
     (expected: `FilterType::Nearest`; confirm by parity test, not by docs);
  3. RGB → CHW float32 → ImageNet normalize.
- `Potatr::forward(&mut self, input) -> (node_logits, boxes, edge_logits)` —
  named-output lookup by substring (`pred_node_logits` etc.), following the
  `scale_0`/`scale_1` lookup pattern in `rapid_table.rs:263-279` (robust to
  session output ordering).

**`src/engine.rs`:** `potatr: Option<Potatr>` slot + `potatr_model_path` +
`ensure_potatr()` lazy loader mirroring `ensure_table` (`engine.rs:382-392`);
provider resolver mirroring `resolve_table_providers` (`engine.rs:148-157`) —
note the table-slot precedent (CPU-pinned, `architecture.md` benchmarks table):
measure before choosing, default CPU until CUDA numbers exist.

**`src/config.rs`:** `TableBackend { Slanet, Potatr, Auto }` on `ModelOpts`
(default `Slanet` in v1 — opt-in; flip to `Auto` only per §9 gate) +
`potatr_node_threshold: f32` (0.8) + `potatr_edge_threshold: f32` (0.5) on
`TextOpts`-adjacent knobs; Python bindings + `.pyi` mirror.

**Test (this phase's gate):** `tests/test_potatr.rs` — preprocess parity:
fixed fixture PNG (reuse `target/page_chem5.png` shape or a checked-in small
fixture) → Rust tensor vs Python reference tensor, exact on NEAREST resize,
1e-6 on normalize. No weights needed — runs in the default suite.

## 5. Phase P1 — postprocess port (medium; the bulk)

Port `postprocess.py` (42 KB) + `metrics.cells_to_html` into `rapid_potatr.rs`
(deterministic geometry, no ML). Function map — port in this order (each is
unit-testable; later ones consume earlier ones):

| # | Python (`postprocess.py`) | Rust | Notes |
|---|---|---|---|
| 1 | bbox primitives (`_bbox_area`, `_bbox_intersect`, `overlaps`, `iob`) | `fn`s on `[f32;4]` | trivial; test with the repo's edge cases if any in `tests/` |
| 2 | `apply_threshold`, `sort_objects_by_score`, sorts | same | — |
| 3 | `extract_text_inside_bbox`, `get_bbox_span_subset`, `extract_text_from_spans` (span/line/block ordering!) | same + token struct `{bbox, text, block, line, span}` | ordering keys are load-bearing; unit-test with synthetic tokens |
| 4 | `refine_rows` / `refine_columns`, `slot_into_containers`, `remove_objects_without_content` | same | needs words; test on chem fixture tokens |
| 5 | `align_columns` / `align_rows` / `align_headers` / `align_supercells`, `header_supercell_tree`, `nms*` family, `rotate_objects_90/270` | same | the fiddly middle; port their `tests/` cases if present, else snapshot ours |
| 6 | `refine_table_structures`, `table_structure_to_cells` | same | cell invariants: `row_nums`/`column_nums`/`header` flags |
| 7 | `potatr_outputs_to_objects` (softmax + threshold + scale to orig) | same | node threshold from config |
| 8 | `potatr_objects_to_table_structures`, `potatr_objects_to_cells`, `potatr_outputs_to_tables_cells` | same | entry points the router calls |
| 9 | `metrics.cells_to_html` | `potatr_cells_to_html` (new; `tables.rs` keeps `html_tables_to_gfm`, this one is POTATR-specific) | mirror the thead/th + tbody/td split exactly (the ikat gate caught a TEDS-0 from getting this wrong) |
| 10 | edge linking (`softmax(edge_logits)[...,1:].max`, bidirectional mean ≥ threshold → `linked_text_parts`) | same | caption/footer association lives here |

Rotated classes (`table rotated`, …): map to the unrotated handling in v1 and
log the label (deskew path is a later proposal if ever).

**Test (this phase's gate):** `tests/test_tex_potatr_parity.rs`, `#[ignore]`,
`BOBINE_POTATR_PARITY=1` (mirrors `test_tex_teller_parity.rs`): load
`potatr.onnx` from the hub cache + run chem-olive p.5 fixture through
preprocess → session → ported postprocess, assert the 21-cell table reproduces
the Python reference byte-identical (cells + HTML). Downloads ~116 MB once;
excluded from the default suite like the TexTeller gate.

## 6. Phase P2 — routing + config + goldens (medium)

**Hook point:** the ONNX-path page loop in `converter.rs` (region dispatch at
`converter.rs:152` / `2654`; confirm which loop owns the ONNX path during
implementation — the pre-pass goes immediately before region-line claiming so
claims are visible to every arm).

**Per qualifying page** (`table_backend != Slanet`, words ≥ `MIN_WORDS`, suggest 10):
1. Build tokens from `page_chars` (`SourceChar` bbox + text + line/block order —
   confirm `SourceChar` carries enough ordering; if not, derive block/line/span
   from the line-claim output `region_lines` + y-sorting).
2. `Potatr::forward` → objects → tables+caption/footer via §5.
3. Emit one markdown block per table (HTML → GFM honoring
   `convert_html_tables`, same as `table_block`'s return path) at visual
   position; caption/footer text appended to the table block (they are
   edge-linked, not free prose).
4. Suppress layout regions with IoU > 0.5 against any claimed box (table,
   caption, footer) by marking them `CLAIM_NONE`-equivalent for this page, so
   neither `table_block` nor the caption/figure arms double-emit. Unclaimed
   regions flow through the existing arms unchanged (SLANet fallback intact).

**Precedence decision (needs your sign-off before code):** POTATR wins inside
its edge-linked claim set; the figure arm keeps all other captions. IoU
threshold 0.5 default, config-exposed.

**Goldens:** chem-olive + econ-tax pages through the new path with
`BOBINE_UPDATE_GOLDENS=1`, hunk-reviewed (office_oxide discipline); corpus
census gains a table column (GT tables vs POTATR tables per paper — the ikat
gate already produced these numbers: 39 GT, ~43 detected).

## 7. Phase P3 — docs + release (small)

- `architecture.md`: table-arm section rewritten (POTATR primary-behind-flag,
  SLANet fallback; §9 review pointer updated from "re-evaluate" to this proposal).
- `benchmarks.md`: rapid-pipeline table gains the POTATR row (340 ms CPU/page
  + page-words cost; CUDA TBD — measure, don't project).
- `CHANGELOG.md` Unreleased entry; `quickref.md` config knobs.
- Thought ingest (`bobine-analysis`): what shipped + measured numbers.
- Version: patch or minor per semver feel — new capability, additive, default
  off → minor bump when the default flips (§9), patch for opt-in landing.

## 8. Test plan (summary)

| Phase | Test | Suite | Gate |
|---|---|---|---|
| P0 | preprocess parity (Rust vs Python tensor) | default (`test_potatr.rs`) | exact / 1e-6 |
| P1 | bbox/text-span units; ported repo cases; 21-cell chem fixture byte-parity | default + ignored (`BOBINE_POTATR_PARITY=1`, ~116 MB weights) | byte-identical |
| P2 | claim/suppress unit (synthetic regions + boxes); golden pages reviewed | default + goldens | reviewer sign-off |
| P3 | in-tree ikat rerun (torch harness already exists in `potatr-export/`) | manual | GriTS-Con ≥ 0.88 (allows render variance) |

## 9. Risks and dependencies

1. **Renderer (blocking for default-on, not for P0–P2).** Production page
   images come from `PdfSource::render_png` (pdf_oxide), CID-corrupt on
   affected papers; the 0.900 gate used clean PyMuPDF renders. Build P0–P2
   behind the render abstraction with clean fixtures; wire the live path only
   where renders are trustworthy. Do not flip the default while P-D numbers
   stay withheld — same rule, same reason.
2. **PIL resample default.** If it is not NEAREST, the Rust filter choice
   changes — the P0 parity test settles it empirically, not this document.
3. **Words quality on scans.** Page-level OCR words cost ≈ one full-page OCR;
   measure and record; gate `Auto` on word density, not just presence, if
   sparse OCR degrades edges.
4. **Threshold transfer.** Node 0.8 / edge 0.5 come from PubTables-2 eval;
   confirm on ikat during P2 (the gate harness makes this a one-flag rerun).
5. **No `*_REV` drift.** Pin `e7f1ce67…` in code from day one (P0); the mirror
   is content-addressed by `manifest.json` sha256 — re-verify on first fetch.

## 10. Effort and order

P0 ~1 day → P1 ~2–3 days → P2 ~2 days → P3 ~½ day; roughly a week, each phase
independently committable with the tree green throughout. Recommended:
P0 → P1, then pause for the §6 precedence sign-off before P2; P3 with the
release that flips the default, not the one that lands opt-in.
