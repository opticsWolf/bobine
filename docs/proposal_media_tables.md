# Proposal: Figures, Images & Tables — Correct-Order Extraction

Status: proposal / plan (not yet implemented)
Scope: bobine Rust core (`converter.rs`, `pdf_source.rs`, `engine.rs`, `config.rs`) + Python bindings
Related: [benchmarks.md](benchmarks.md) (measured CPU/CUDA numbers), [architecture.md](architecture.md)

---

## 1. Problem statement

### Figures / images today
| Path | Behaviour | Defect |
|---|---|---|
| Embedded-image pre-pass (`extract_image_files`, all modes) | dumps every embedded raster to `p{n}_img{k}.png` | no placement info used; logos/decoration rules included |
| Fast path (`fast_page_markdown`) | pdf_oxide `to_markdown` with `include_images:false` | zero figures in output |
| Gallery (`maybe_append_gallery`) | appends unreferenced `p{n}_img*` at page end | wrong position, no captions, rasters only |
| ALWAYS/AUTO figure regions | render-crop of the 300-dpi page render → `_reg_fig_{k}.png`, emitted at region position | good position, but: text inside figures is baked into PNG (never searchable), overlapping figure boxes produce duplicate near-identical crops, vector-only vs raster content not distinguished |

Net effect: the same figure can appear twice (crop + gallery) or not at all (fast path), never at its true reading position in the fast path, and figure text is lost everywhere.

### Tables today
- **Fast path**: pdf_oxide `to_markdown` already detects born-digital ruled tables (`ConversionOptions.extract_tables: true` by default) — this works.
- **ONNX path, born-digital `"table"` regions**: `region_lines_to_text()` plain-text dump run through `html_tables_to_gfm` (a no-op on non-HTML) → **all cell structure lost**, even though pdf_oxide exposes exactly the needed API.
- **Scanned tables**: OCR crop → SLANet-plus → HTML → GFM. Works.

### Key pdf_oxide capabilities currently unused
| Capability | API | Since |
|---|---|---|
| Embedded images **with bbox in PDF user space**, rotation, CTM | `PdfImage::bbox()/matrix()`, `extract_images_in_rect()` | 0.3.14 |
| Structured table extraction (StructureTree + spatial grid, rowspan/colspan, prose rejection) | `extract_tables_in_rect(page, rect[, config])` → `Vec<Table>` | 0.3.14 |
| Table bbox for excluding table spans from flow text | `Table.bbox` | — |
| Markdown converter image emission (opt-in, file or base64) | `ConversionOptions.include_images/image_output_dir` | — |
| Column-aware XY-Cut reading order | `ReadingOrderMode::ColumnAware` (default fallback) | — |

---

## 2. Design proposals

### P-A. Unified per-page media inventory (foundation)

Introduce a cheap, ML-free per-page record built once in `convert_pdf_source`:

```rust
struct MediaRecord {
    kind: MediaKind,          // Raster | VectorRegion | TableGrid
    bbox_pts: Rect,           // PDF points (user space)
    source: MediaSource,      // Embedded(PdfImage) | RenderCrop
    asset_path: PathBuf,      // written PNG
    confidence: f32,
}
```

Sources merged per page:
1. **Embedded rasters** — `extract_images()`; each carries `bbox()` → keep only *content* images: filter out decorations by rendered-area threshold (`min_figure_area_pts`, new config, default ~100 pt²) and by repeat-count (same XObject drawn on ≥ 50% of pages ⇒ running logo/header art).
2. **Layout figure regions** (ALWAYS/AUTO only) — existing DocLayout-YOLO `figure`/`image` regions.
3. **Detected grids** — `extract_tables_in_rect` hits (see P-C).

**Dedup rule:** an embedded raster whose bbox overlaps a figure region with IoU ≥ 0.5 wins (original bytes, correct resolution, no baked-in neighbours); the render-crop is discarded. Crops remain only for vector/mixed content with no underlying raster. Overlapping figure regions themselves are deduped by keeping the highest-confidence box per IoU cluster.

### P-B. True reading-order placement (kills the gallery)

Replace `maybe_append_gallery` with **stream interleaving**:

- In the ALWAYS/AUTO path, media records are emitted as `![caption](path)` blocks at their region's position in the existing reading-order block stream (already correct — this is where figure crops land today; the change is that they now prefer the deduped embedded asset).
- In the fast path, build the same stream without ML: text lines (from `chars()`, reusing `cluster_lines()` + gutter splitting) and media records are merged into one list sorted by `(y_band, x)`. Each text line knows its y; each media record its bbox. Output = markdown walking this list. The gallery disappears entirely.
- Caption association: lines matching `^(Figure|Fig\.|Table|Abb\.|图)\s*\d*[:.]?` within ~2 line-heights below/above a media record are attached as the italic caption line under the image instead of flowing as body text.

Config additions (must be mirrored in Python — see §3 Phase 4):
```toml
[images]
mode = "link"            # "link" (default) | "embed_base64"
output_dir = "assets"    # structured relative dir for mode=link
max_embed_bytes = 262144 # cap for embed_base64; larger figures stay links
attach_captions = true
min_figure_area_pts = 100 # drop decorations
figure_crop_dpi = 150     # cap for render crops
```

**Reference/embedding policy — decision:** prefer **B, structured relative paths**
(`assets/page-1/fig-2.png`); plain bare filenames (A) remain the fallback shape.
Base64 data URIs are opt-in only (`mode = "embed_base64"`) with a size cap,
because a 300-dpi render crop can be megabytes and data URIs bloat/diff badly.
Inline HTML `<img>` is rejected (not pure GFM). Captions are emitted both as
alt text (`![Figure 1: …](path)`) and as the italic line below the image.
Downstream, the pipeline's asset-store stage keeps rewriting these links to
`okf-asset://` ids, so the converter only needs stable, page-scoped names:
`assets/p{n}/fig-{k}.{ext}`.

### P-C. Tables: structured-first cascade

Rewrite the `"table"` region branch as a three-stage cascade:

1. **Born-digital structured** — `pdf.extract_tables_in_rect(index, bbox)`:
   - hit → render GFM directly from `Table.rows` (extend `tables.rs` with `table_to_gfm(&Table)` honoring `rowspan`/`colspan`: cells with spans >1 keep the whole table as HTML, matching current `html_tables_to_gfm` behaviour).
   - register `Table.bbox` in the page's claimed-rect set so flow text does not re-emit cell contents.
2. **Born-digital unstructured** — current glyph-line dump (fallback for borderless tables the spatial detector rejects).
3. **Scanned** — existing OCR → SLANet-plus → HTML → GFM path.

Fast path needs no change (pdf_oxide already emits markdown tables there).

### P-D. Layout-preservation upgrades (incremental)

1. **Seam repair instead of trailing dump** — unowned glyphs (currently appended as a trailing block) are offered to the *nearest* region by edge distance before falling back to the trailing block. Removes most "fragment appended at end" artifacts introduced with glyph ownership.
2. **Figure-aware column model** — media records participate in the gutter analysis of `cluster_lines()` so a figure between columns doesn't fuse the two columns' lines.
3. **Optional strict mode** — pass-through `preserve_layout`-style flag to pdf_oxide for documents where semantic reflow is unwanted (off by default).

Non-goals: pixel-perfect layout reproduction; OCR of figure interiors (kept as future work — pdf_oxide's gated `ocr` feature / Auto extractor could feed figure-interior text into reading order later).

---

## 3. Implementation plan

### Phase 1 — Born-digital tables (smallest, highest value) · ~1 day — **DONE**
- Added `SourceTable/Row/Cell` (bobine-owned mirrors) + `PdfSource::tables_in_rect`
  (default: empty; real impl: whole-page `extract_tables` with default Both/Both
  config + bbox intersection filter).
- `tables.rs`: `source_table_markdown()` (GFM pipe table; merged cells → HTML),
  `is_plausible_table()` acceptance gate.
- Converter table branch is now the 3-stage cascade (P-C); gated by new
  `ConverterConfig::structured_tables` (default true, mirrored in Python config).
- Fixture `tests/fixtures/ruled_table.pdf` via `examples/make_table_fixture.rs`;
  probes `probe_tables.rs`, `probe_paths.rs`. 14 new unit tests.

**Phase 1 findings (measured):**
- ⚠️ `pdf_oxide::api`'s `extract_tables_in_rect` silently applies the **relaxed**
  text-only strategy — it misses ruled grids. bobine therefore extracts
  whole-page with the balanced default config and filters by bbox itself.
- ⚠️ Detector quality on real output is mixed:
  - arXiv 1706.03762 (booktabs tables): fragmented grids, missing headers,
    half-empty rows → rejected by the 60%-fill gate → falls back to text dump.
  - Synthetic ruled grid (`ruled_table.pdf`): prose sliced into columns and
    body rows concatenated into one cell → also gated out.
  - Net effect: stage 1 currently fires only on detector-friendly grids and is
    **safe by construction** elsewhere. Revisit when pdf_oxide's
    `spatial_table_detector` improves; no redesign needed on our side.

### Phase 2 — Media inventory + dedup · ~1–2 days — **DONE**
- `PdfSource::images()` → `SourceImage { bbox, width, height }` (bobine-owned,
  default-empty like `tables_in_rect`); fakes support `with_image_at()`.
- Per-page `EmbeddedAsset` inventory built once in the conversion loop and
  threaded through routing (`route_page` → `full_structure_page_markdown`) and
  the gallery.
- Figure branch prefers the ORIGINAL embedded raster when the region covers ≥60%
  of it — COVERAGE, not IoU: DocLayout boxes are routinely coarser/offset from
  true bitmap placement, so symmetric overlap almost never fires (measured on
  arXiv 2608.05540). Crops remain for vector/mixed art.
- Crop filenames now page-scoped (`_reg_p{n}_fig_{k}.png`) — previously every
  page's first figure overwrote `_reg_fig_0.png` in the shared work dir, so
  earlier pages' image links silently pointed at the wrong figure.
- `dedup_figure_regions()`: overlapping (IoU ≥ 0.5) duplicate figure boxes
  dropped before glyph ownership; text/table regions unaffected.
- Gallery rewritten per-asset: unreferenced assets are appended individually
  (previously ANY inline `![` suppressed the whole gallery, losing figures the
  layout model missed); decoration-sized placements filtered via new
  `min_figure_area_pts` config (default 100 pt², mirrored to Python).
- **Measured (2608.05540):** 9 image refs, all unique; 3 figures served from
  original embedded rasters; solitons regression clean (6 unique refs).
- Tests: rect_iou, coverage match, figure dedup, per-asset gallery.

### Phase 3 — Reading-order interleaving (gallery removal) · ~2 days — **DONE**
- Asset writer: structured naming `<image_output_dir>/p{n}/img{k}.{ext}`;
  JPEG-encoded embeds are written byte-exact (`.jpg`), everything else
  transcoded to lossless PNG. Figure crops join the tree as
  `assets/p{n}/crop{k}.png`.
- New `image_output_dir` config (default "assets", mirrored to Python).
- Fast-path interleaving (`interleave_images`): text-layer lines via
  `cluster_lines`, each content-sized asset anchored before the first line
  below its bbox bottom, located in the markdown by monotone whitespace-
  tolerant substring search (full line → 8 words → 4 words). Unmatchable or
  unplaced assets fall back to the tail gallery (`maybe_append_gallery`),
  which remains the single tail authority — no double emission.
- All fast/surgical return paths now interleave; markdown links are
  work-dir-relative with forward slashes (`assets/p22/img1.jpg`).
- **Measured (2608.05540, NEVER mode):** 5 refs / 5 unique, each landing
  directly beside its figure caption; conversion still ~336 ms for 28 pages.
  ALWAYS-mode solitons regression clean.
- Tests: interleave-between-bands integration test; gallery/structured-path
  assertions updated.

**Follow-up (implemented same day): caption alt-text emission.**
- `caption_alt()` recognises "Figure/Fig./Table/Abb./Plate N…" lines,
  collapses whitespace, truncates at word boundaries (~120 chars), strips
  square brackets (alt-syntax safety); anything else → empty alt.
- Fast path: a window of text lines straddling the asset's bbox bottom is
  scanned (rasters often include padding, so the caption may sit slightly
  ABOVE the bottom edge); the closest caption-pattern line wins.
  Measured on arXiv 2608.05540: 3 of 5 figures get real alt text, the rest
  safely stay empty.
- ALWAYS/AUTO path: `nearby_caption_text()` deterministically associates the
  nearest `caption`-labelled layout region (vertical gap ≤ 3 median line
  heights, ≥ 30% horizontal overlap).
- Known limit: side-by-side caption placement in multi-column layouts can
  fall outside the vertical-gap check → empty alt (safe fallback).

**Still deferred:**
- `embed_base64` image mode + `max_embed_bytes` cap (no base64 crate in deps).
- Side-by-side caption association for multi-column layouts.
- **Tests:** synthetic page with image bbox between two text bands → image lands between them in output; caption attaches.
- **Accept:** gallery gone; images appear at their visual position in every mode.

### Phase 4 — Polish, headings & plumbing · ~1–2 days **DONE**
- **Heading-promotion post-pass (new):** scholarly section patterns → markdown
  headings, applied to the final page markdown in every mode:
  - `I. INTRODUCTION` / `II. RELATED WORK` → `## `
  - `3.1 Encoder and Decoder Stacks` / `A. Setup` → `### `
  - `3.1.1 …` → `#### `
  Guarded strictly to avoid prose false positives: numbered/roman prefix
  pattern + short line (< ~100 chars) + mostly-uppercase or existing bold
  markers (`**…**`) already emitted by pdf_oxide. Rationale: measured on our
  fixtures pdf_oxide's `detect_headings` only catches display titles
  (`## Attention Is All You Need`) and misses numbered sections entirely;
  solitons' title is missed too; bobine's ONNX `title` label gives a single
  `##` level with no hierarchy.
  - Also promote a detected document title to `# ` when it is the first
  block of the document and no heading exists yet.
- Seam-repair upgrade (P-D.1): unowned glyphs offered to the nearest region
  before the trailing-block fallback.
- Side-by-side caption association for multi-column layouts (extends the
  alt-text work's vertical-gap check with horizontal adjacency).
- Update `docs/architecture.md` + `docs/benchmarks.md`; rerun full suite +
  `e2e_auto` on all fixtures; add golden files for figure-heavy pages.
- **Accept:** both fixture papers yield a navigable outline (`grep '^#'`) with
  title as H1 and sections/subsections at H2/H3; no false-positive headings in
  body prose (golden-diff verified).

### Risks / open questions
- `PdfImage.bbox` is `Option` — some XObjects may lack placement; those degrade to end-of-page gallery entries (keep a minimal fallback).
- `extract_tables_in_rect` prose-rejection may miss borderless tables → cascade stage 2 covers it.
- Golden-test churn: fast-path output changes wherever tables/figures exist.
- Performance: `extract_paths`/table detection adds work per page; measure with `bench_rapid`-style timing on the arXiv fixtures before/after Phase 1.
