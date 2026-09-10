# Office export plan — md for all formats, csv/json for Excel, pictures everywhere

Closes the gaps from the Office-path review (`docs/code_review.md` era findings):
today `convert_office` is 6 lines (`Document::open` + `to_markdown`) with zero
real coverage, embedded pictures are silently dropped, and spreadsheets only
yield one opaque markdown string. All building blocks below were verified
against `office_oxide 0.1.8` (locked in `Cargo.lock`).

## Conventions (repo standing rules)

- Patch version bump per phase (`Cargo.toml` + `pyproject.toml` + `Cargo.lock`).
- Commit + push per phase; tree clean before the next.
- Live-verify where models/IO are involved; unit tests for pure logic.
- `cargo check` warning-free (baseline: pre-existing warnings only).

## Phase 0 — Fixtures + harness (real coverage first)

**Why first:** neither implementation ever verified real Office output (Rust: no
tests at all; legacy: `FakeOffice` mock only). Everything below needs fixtures.

- Generate fixtures with office_oxide's own `create` API (no binaries in git
  from unknown sources):
  - `tests/fixtures/office/basic.docx` — headings 1–3, paragraph styles, bold/
    italic, bullet + numbered lists, a 3×3 table (header row, merged cell),
    hyperlink, footnote, one embedded PNG with alt text.
  - `tests/fixtures/office/types.xlsx` — 2 sheets (`Data`, `Summary`); string /
    int / float / bool / date / percent / currency cells, one formula cell
    (`=SUM(...)`), one merged-cells header, one fully-empty row, unicode text.
  - `tests/fixtures/office/deck.pptx` — title slide, bullets, one table, one
    embedded PNG.
  - Legacy (if `create` supports them; else check in minimal hand-built files
    and document provenance): `legacy.doc`, `legacy.xls` (2 sheets, mixed
    types), `legacy.ppt`.
- **Outcome (v0.5.1): `create_from_ir` supports Docx/Xlsx/Pptx only — no
  legacy writers exist in 0.1.8, so no legacy fixtures. Deferred until
  real-world samples are available; `tests/test_office.rs` takes new
  fixtures with zero harness changes.**
- Harness: `tests/test_office.rs` with helpers `assert_contains_in_order(md,
  &[…])`. Gate 1 test per format: convert succeeds, key content present in
  order, no `panic!` on any fixture (fuzz-adjacent smoke).
- Acceptance: `cargo test --test test_office` green; fixtures < 200 KB total.

## Phase 1 — Markdown export verified per format

- For each of the 6 formats, assert the md preserves: headings (`#` levels),
  lists, tables (GFM pipes via existing `html_tables_to_gfm` path if the IR
  renders HTML tables — verify, don't assume), image *references* (placeholder
  until Phase 3), sheet/slide boundaries (`---` or `## Sheet: <name>` /
  `## Slide N` — decide, document in quickref).
- Fix bobine-side issues found; file office_oxide issues upstream with minimal
  repros instead of working around them in-tree (no vendoring).
- Drive-bys from the review: `convert_office(&self)` → associated function
  (uses no `self`/config/work_dir); fix legacy `convert_office` docstring
  ("DOCX/XLSX/PPTX" → all six).
- **Outcome (v0.5.2): both drive-bys done. Upstream rendering gaps logged
  for office_oxide (no in-tree workarounds): footnote bodies dropped,
  hyperlink URLs dropped (text kept), formula cells render empty, pptx
  bullets lose `-` markers, pptx tables are TSV not GFM, pptx images
  dropped silently. `tests/test_office.rs` pins the must-hold subset
  (GFM tables docx/xlsx, sheet/slide `##` boundaries, typed
  date/percent cells, image alt text).**
- Acceptance: per-format assertions green; no `#[ignore]` left behind.

## Phase 2 — Excel multi-format export (csv / json / md)

**API surface (new, additive — `convert_office` keeps working):**

```rust
pub struct SheetData {
    pub name: String,
    pub headers: Vec<String>,        // first non-empty row, may be empty
    pub rows: Vec<Vec<CellData>>,    // strings post-format, rectangularized
}
pub struct CellData {
    pub text: String,                // display text (number/date formatting applied)
    pub raw: CellJson,               // typed value for JSON
    pub formula: Option<String>,     // e.g. "=SUM(A1:A5)", when present
}
pub struct ExcelDocument {
    pub sheets: Vec<SheetData>,
    pub markdown: String,            // "## Sheet: <name>" sections + GFM tables
}
pub fn convert_excel(path: &Path) -> Result<ExcelDocument>;
pub fn sheets_to_csv(doc: &ExcelDocument) -> Vec<(String, String)>;  // (sheet, csv)
pub fn excel_to_json(doc: &ExcelDocument) -> serde_json::Value;
```

- **Cell access (verified present):**
  - xlsx: `XlsxDocument.workbook.sheets` (names) + `worksheets[].rows[].cells[]`
    (`Cell { reference, value: CellValue, style_index, formula }`) +
    `format_cell_value(&cell)` for display text. Reuse `sheet_to_csv` output
    where byte-identical to ours (test it; don't duplicate quoting logic).
  - xls: `XlsDocument.sheets[].{name, rows: Vec<Vec<CellValue>>}` — render
    csv/md/json in bobine (no `sheet_to_csv` equivalent in 0.1.8).
- **Spike first (½ day box):** enumerate `CellValue` variants in 0.1.8 and pin
  the JSON mapping: numbers (int vs float — preserve `1` vs `1.0`?), dates
  (ISO-8601 string + Excel serial raw?), bools, errors (`#DIV/0!` → string +
  `null` raw?), empty (skip trailing empties per row? rectangularize to max
  width with `""`?). Formulas: emit cached `text`, expose `formula` field,
  never evaluate. Merged cells: top-left value, rest empty (document it).
- **Pipeline integration:** `ingest_document` on `.xls/.xlsx` writes
  `<stem>.md` (as today) **plus** `<stem>.<sheet>.csv` siblings and
  `<stem>.json` into the output dir; extend `ConvertedDocument` with
  `data_files: Vec<PathBuf>` (additive field). Python: `convert_excel(path)`
  returning an object with `.markdown` / `.csv_by_sheet` / `.json`.
- **Deps:** promote `serde_json` from `[dev-dependencies]` to `[dependencies]`;
  add `csv = "1"` (correct RFC-4180 quoting; don't hand-roll).
- Acceptance: `types.xlsx` + `legacy.xls` → csv parses under Python `csv`
  module; json schema asserted field-by-field in tests; md contains one GFM
  table per non-empty sheet with `## Sheet:` headers; empty sheets/rows don't
  crash and don't emit phantom tables.

## Phase 3 — Pictures exported/extracted

**Verified present:** IR `Element::Image(Image { data: Option<Vec<u8>>,
format: Option<ImageFormat>, alt_text, decorative, positioning })`, populated
by `convert_docx` (2 sites), `convert_xlsx` (drawings), legacy `doc/images.rs`
+ `XlsImage` exist. `Document::to_ir()` dispatches all six formats.

- Walk `doc.to_ir()` sections/elements in bobine (new `src/office_images.rs`
  or extend `assets.rs` — decide at implementation; reuse wins):
  - `data: Some(bytes)` → stage exactly like PDF figures: content-hash filename
    (`<image_output_dir>/p{page|sheet}/img{k}.{ext}` — Office has no pages;
    use `doc/` scope or sheet/slide index), link `![alt](rel)` at the
    element's position in the markdown (docx inline order; pptx slide order;
    xlsx: after the owning sheet's table).
  - `format` → extension mapping (`ImageFormat` → `png/jpg/...`; default
    `bin` + warn when unknown — never guess pixels).
  - `decorative == true` → exclude from gallery (mirrors `min_figure_area_pts`
    decoration rule for PDFs).
  - `data: None` (linked, not embedded) → keep alt text if present, else drop
    silently (log at debug). Never emit broken links.
- Spikes (time-boxed, record outcomes in this file): (a) do legacy
  doc/xls/ppt `to_ir()` paths emit `Image` with `data`? (`XlsDocument.images`
  is private — IR path may or may not surface it); (b) pptx floating vs
  inline positioning fidelity. If office_oxide gaps block us: minimal upstream
  PR first, fallback documented in-plan (no in-tree forks).
- `extract_images: false` must suppress staging (respect the knob — today it
  can't, since nothing is staged; don't regress the knob's meaning).
- Acceptance: docx/xlsx/pptx fixtures round-trip their PNGs byte-identical
  (hash compare staged file vs source); md links resolve to staged files;
  decorative images excluded; `data: None` case covered by a linked-image
  fixture; legacy formats: whatever the spikes prove, asserted.

## Phase 4 — Bindings, pipeline UX, docs

- PyO3 (`src/py_bindings.rs`, `python/bobine/__init__.pyi`): `convert_excel`,
  image-inclusive office md (no signature breaks — additive only).
- `docs/quickref.md`: Office section (formats × outputs matrix, csv/json
  siblings, image staging, `BOBINE_CACHE_DIR` N/A note — Office needs no
  models). `docs/architecture.md`: office data-flow paragraph.
- Goldens: `tests/golden/office/*.md` for the Phase 0 fixtures (fast,
  deterministic — office path needs no models).
- Legacy: port the `convert_excel` equivalent? No — legacy is frozen reference
  (`legacy/`). Only fix its `convert_office` docstring (Phase 1 drive-by).
- Acceptance: `cargo test` full suite green incl. new `test_office` +
  `test_excel`; `maturin develop` smoke on the Python side; docs build clean.

## Test matrix (acceptance summary)

| Input | md | csv/sheet | json | images |
|---|---|---|---|---|
| basic.docx | ✓ ordered | n/a | n/a | byte-identical + alt |
| types.xlsx (2 sheets, types, formula, merged, empty row) | ✓ per-sheet tables | ✓ parses | ✓ schema | n/a (add pic in Phase 3 fixture) |
| deck.pptx | ✓ slide order | n/a | n/a | byte-identical |
| legacy.doc/.xls/.ppt | ✓ no-panic + content | xls ✓ | xls ✓ | per spike outcomes |

## Risks

1. **office_oxide 0.1 churn** — pre-1.0 dep with huge surface; pin exact in
   `Cargo.lock` (already), upstream-first for fixes, no vendoring.
2. **Legacy fidelity** (doc/xls/ppt) — CFB-era formats; scope is
   content-preserving md + xls data, not pixel fidelity. Spike outcomes gate
   Phase 3 legacy rows.
3. **Formula JSON** — cached values only, never evaluate (documented above).
4. **Image positioning in xlsx** (drawings float over cells) — sheet-level
   placement after the table is the sane default; cell-anchored II placement
   is explicitly out of scope.
