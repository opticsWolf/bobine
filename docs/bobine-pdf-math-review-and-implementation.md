# Bobine PDF-to-Math Review and Implementation Plan

**Scope:** extract mathematical formulas from PDFs using semantic PDF data before OCR, while preserving Bobine's Rust API, Python surface, and existing Markdown behavior.

- **Target checkout:** `D:\User\Documents\Rust\bobine`
- **Current branch:** `dev_0.10.0`
- **Bobine version:** `0.10.0`
- **Locked PDF backend:** `pdf_oxide 0.3.78` (`Cargo.lock`)

**Backend decision:** use `pdf_oxide` for both tagged structure-tree parsing and embedded-file extraction. Do not add `lopdf` or another PDF parser. First integrate the existing `pdf_oxide::document::PdfDocument` APIs with Bobine's `PdfSource` seam. Use the lower-level object accessors exposed by `PdfDocument` only for PDF fields that the parsed structure model does not retain.

---

## 1. Executive summary

Bobine already has the main OCR and output pipeline needed for formula conversion:

- math-font and Unicode math detection;
- formula bounding boxes and multi-line clustering;
- text-layer and layout-based formula localization;
- TexTeller ONNX recognition, LaTeX sanitization, and plausibility checks;
- inline and display math Markdown;
- MathML-to-LaTeX conversion in `mml.rs`;
- JATS-to-Markdown conversion in `jats.rs`.

The missing bridge is between PDF semantics and those existing math converters. Bobine's `PdfSource` currently wraps `pdf_oxide::api::Pdf`, and it has no math or attachment methods. The locked `pdf_oxide 0.3.78` already provides:

- `PdfDocument::structure_tree()` and `structure_tree_with_budget()`;
- public tagged structure types including `StructTreeRoot`, `StructElem`, `StructChild`, `StructType::Formula`, page references, and marked-content references;
- `PdfDocument::extract_embedded_files()` for catalog `/Names /EmbeddedFiles` attachments.

The implementation should integrate these capabilities rather than parse each PDF again with another crate.

The locked 0.3.78 API audit is resolved as follows:

1. The parser skips `/A` and `/Alt` and does not retain the source object reference. Bobine can inspect those fields through the same document's raw object APIs, but must align a narrow formula lookup to the projected tree.
2. `/AF` is readable from raw Formula dictionaries. `extract_embedded_files()` supplies bytes only for files also listed under `/Names /EmbeddedFiles`, and it omits filespec identity and metadata; match a unique filename and skip ambiguous/unlisted associations.
3. `extract_embedded_files()` has no per-call decompression bound and decodes before Bobine's size check. Treat `max_embedded_bytes` as a downstream parsing limit, not a peak-memory limit.
4. A structure element has page and marked-content references, but no direct geometry. Map MCIDs to page text positions using `pdf_oxide` data where available; leave the bbox absent when the mapping is ambiguous.

Keep all PDF object parsing and stream decoding in `pdf_oxide`. If the narrow raw lookup cannot be aligned safely with its parsed projection, skip the association and report it; do not introduce another parser.

### Recommended order

1. Establish the `pdf_oxide` API and fixture baseline.
2. Add math provenance types and no-op `PdfSource` methods.
3. Integrate `PdfDocument` with Bobine's PDF source adapter.
4. Read `/Formula` structure elements and MathML, with structure MathML preferred over formula-associated MathML and `/Alt`.
5. Extract document attachments with `extract_embedded_files()`; handle formula-level `/AF` separately.
6. Route structured spans before OCR, suppressing only OCR boxes they cover.
7. Add validation, reporting, documentation, and Python bindings.
8. Benchmark structured extraction; defer HTML export.

---

## 2. Why this matters

PDF pages are designed for visual presentation. Formula meaning often depends on two-dimensional placement, and text extraction can flatten superscripts, subscripts, and fractions. Tagged PDFs can provide a logical structure tree, and accessible math may be represented with `Formula` structure elements and MathML. MathML and JATS are preferable inputs when available because Bobine already has a MathML-to-LaTeX converter and a JATS route.

Bobine's existing JATS/MathML work demonstrates the value of structured math: source formulas avoid OCR errors and can be converted on CPU. This plan adds a PDF-side semantic path while retaining the current PDF/OCR path for untagged documents.

---

## 3. Current-state audit

### 3.1 Relevant Bobine modules

| Module | Current role |
|---|---|
| `src/pdf_source.rs` | `PdfSource` seam and implementation for `pdf_oxide::api::Pdf` |
| `src/converter.rs` | page routing, math box detection, OCR, Markdown placement |
| `src/mml.rs` | MathML-to-LaTeX conversion |
| `src/jats.rs` | JATS XML-to-Markdown conversion |
| `src/sanitize.rs` | LaTeX repairs and brace-balance checks |
| `src/config.rs` | flattened, serde-compatible converter configuration |
| `src/py_bindings.rs`, `python/bobine` | Python configuration and public API |

### 3.2 Existing math behavior

- `math_boxes_from_chars()` detects candidate formulas from text-layer glyphs.
- `cluster_math_boxes()` groups fragments for fewer, better OCR crops.
- `route_page_inner()` selects fast, surgical, or full-structure processing.
- `sanitize_latex()` and `plausible_display_latex()` filter OCR output.
- `mml_to_tex()` is already available for converting MathML nodes.
- `PdfSource::tables_in_rect()` shows how Bobine can expose structured PDF information to the converter without coupling converter logic to a backend.

### 3.3 Baseline gaps before this implementation

| Gap | Current state |
|---|---|
| Tagged math | `PdfSource` has no structured-math method; converter does not read `StructTreeRoot` |
| Formula-level MathML attachment | no `/AF` association is surfaced to Bobine |
| Document attachments | no `PdfSource` method exposes `extract_embedded_files()` |
| Source-first routing | structured formula spans do not precede ONNX routing |
| Provenance | callers cannot distinguish tagged MathML from OCR output |
| Validation | current checks focus on OCR output and the version-scoped render-health gate |

### 3.4 What the locked `pdf_oxide` API provides

Use the API in the locked version, not assumptions based on a newer docs.rs release:

- `pdf_oxide::document::PdfDocument::structure_tree()` returns an optional `StructTreeRoot`.
- `structure_tree_with_budget()` allows a parse-time budget.
- `StructTreeRoot` exposes root elements, the parent tree, and role map.
- `StructElem` exposes structure type, children, page, actual text, original role, and fields for attributes/Alt/Expansion. In 0.3.78, however, the parser explicitly skips `/A` and `/Alt`; those fields remain empty, and the parsed element does not retain its source `ObjectRef`.
- `StructChild` distinguishes nested structure elements, marked-content references (MCID, page, scope), and object references.
- `PdfDocument::extract_embedded_files()` returns `Vec<(String, Vec<u8>)>` from the catalog's embedded-files name tree.
- `PdfDocument::catalog()`, `resolve_object()`, and `load_object()` expose raw catalog objects and indirect references. These APIs are sufficient for a narrow `/K` lookup of formula dictionaries and their `/AF`, `/A`, `/Alt`, and `/NS` entries; they do not provide a ready-made formula-to-raw-object association.
- `extract_embedded_files()` returns only filenames and decoded bytes. It traverses `/Names /EmbeddedFiles`, requires an authenticated document, and uses pdf_oxide's encryption-aware stream decoder. It does not return the original filespec identity, `/AFRelationship`, or the embedded stream `/Subtype`.
- `Object::decode_stream_data()` and `decode_stream_with_options()` do not combine the document's encryption handler with per-call decompression limits. Do not use either as a replacement for `extract_embedded_files()` on raw `/AF` streams.

The existing Bobine adapter imports `pdf_oxide::api::Pdf`. The implementation should decide whether Bobine can use `PdfDocument` directly for all required `PdfSource` operations, or needs a small adapter that shares or owns the relevant `pdf_oxide` document state. Avoid opening and parsing the PDF twice if the public API permits a single document instance.

---

## 4. Target architecture

### 4.1 Source precedence

For each page or formula region, use:

1. MathML structure elements below a tagged `/Formula`.
2. MathML associated with that formula through `/AF`.
3. Supported embedded MathML/JATS/XML source, subject to document-level routing rules.
4. `/Alt` as an accessibility description only; never treat it as LaTeX.
5. Existing text-layer extraction and OCR.

If a page contains both tagged and untagged formulas, use structured data for the tagged spans and retain OCR for the remaining formula regions.

### 4.2 Provenance types

Add a small Rust module such as `src/math_source.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathOrigin {
    TaggedMathMl,
    AssociatedFile,
    EmbeddedSource,
    OcrTexTeller,
    TextLayer,
    AltDescription,
}

#[derive(Debug, Clone)]
pub struct MathSpan {
    pub page: usize,           // zero-based
    pub bbox: Option<Rect>,    // PDF points; absent when not reliably known
    pub latex: String,         // bare LaTeX; Markdown fences are added later
    pub display: bool,
    pub origin: MathOrigin,
    pub confidence: f32,
    pub tag: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct MathSourceReport {
    pub tagged_spans: usize,
    pub associated_file_spans: usize,
    pub embedded_source_spans: usize,
    pub ocr_spans: usize,
    pub text_layer_spans: usize,
    pub alt_only_spans: usize,
    pub pages_with_structured_math: usize,
    pub suppressed_ocr_spans: usize,
    pub validation_failed: usize,
    pub validation_repaired: usize,
    pub validation_skipped: usize,
}
```

Keep LaTeX bare internally. Define confidence policy separately from origin: a structured origin can still receive a warning or reduced confidence if parsing or validation is incomplete.

### 4.3 Extend `PdfSource`

Mirror the existing default-method pattern used for `tables_in_rect()`:

```rust
fn structured_math(&mut self, _index: usize) -> Result<Vec<MathSpan>> {
    Ok(Vec::new())
}

fn embedded_files(&mut self) -> Result<Vec<EmbeddedFile>> {
    Ok(Vec::new())
}
```

The Bobine-owned `EmbeddedFile` should preserve filename and bytes. Add MIME type and relationship only when those values are available from the `pdf_oxide` API path used; do not infer an `/AF` relationship from a filename extension.

Generic and fake `PdfSource` implementations keep their current behavior through empty defaults.

### 4.4 PDF adapter and configuration

Prefer a `PdfSource` implementation backed by one `pdf_oxide::document::PdfDocument` if it can supply Bobine's existing text, image, render, and table methods. If the current `api::Pdf` wrapper is required, add a narrow Bobine adapter around the `pdf_oxide` API and document any unavoidable second open.

Add a flattened `MathOpts` group to `ConverterConfig` with serde defaults:

- `prefer_structured_math: bool`;
- `allow_embedded_sources: bool`;
- `max_embedded_bytes: usize` (per document-level embedded source file; 16 MiB default);
- `max_formula_attachment_bytes: usize` (per formula-associated MathML attachment; 1 MiB default);
- `max_struct_nodes: usize` (Bobine-side traversal budget);
- `validate_with_katex: bool` (default off, optional feature);
- `emit_provenance_comments: bool` (default off).

Check `pdf_oxide`'s parser and stream-decompression limits before claiming these values bound the work performed inside the backend. A post-extraction size check protects downstream routing, but does not bound memory already used to decode a stream.

---

## 5. Stepwise implementation phases

### Phase 0 — baseline and `pdf_oxide` API audit

**Goal:** make the backend integration concrete before changing routing.

1. Confirm the branch, locked Bobine and `pdf_oxide` versions, and current local changes.
2. Inventory existing MathML, JATS, PDF, and golden fixtures.
3. Read the locked `pdf_oxide` source/API for:
   - `PdfDocument::structure_tree()` and `structure_tree_with_budget()`;
   - structure tree types and how `/K`, role maps, MCIDs, `/Pg`, `/ActualText`, `/Alt`, `/NS`, and `/A` are represented;
   - `extract_embedded_files()`, filename decoding, stream limits, and whether `/AF` is exposed;
   - opening and sharing document state with the existing `api::Pdf` adapter.
4. Write down which values are available from the parsed tree and which require `PdfDocument` object accessors.
5. Prepare synthetic fixtures containing structure-only MathML, attachment-only MathML, both sources, Alt-only, and malformed/cyclic structures. Prefer a `pdf_oxide` writer API if it can create the required tagged structures; otherwise use small checked-in synthetic PDFs. Do not add third-party copyrighted PDFs or a second parser as a fixture dependency.

**Acceptance:** the chosen adapter design uses the locked `pdf_oxide` API, and any missing API capability is identified before converter routing work begins.

### Phase 1 — provenance and no-op plumbing

**Goal:** add public types and seams with no behavior change.

1. Define `MathOrigin`, `MathSpan`, `MathSourceReport`, and `EmbeddedFile`.
2. Add default `structured_math()` and `embedded_files()` methods to `PdfSource`.
3. Add `MathOpts` with defaults that preserve existing output and flatten it into `ConverterConfig`.
4. Re-export Rust types and expose a read-only `math_source_report()` on `HybridConverter`.
5. Update PyO3 configuration, Python exports, type stubs, and configuration documentation in the same change.

**Acceptance:** old config files still deserialize, fake sources return empty values by default, and untagged document output remains unchanged.

### Phase 2 — adapt Bobine to `PdfDocument`

**Goal:** make `pdf_oxide` structure and attachment methods reachable through `PdfSource`.

1. Implement or adapt the concrete PDF source so it can call `structure_tree()` and `extract_embedded_files()`.
2. Preserve current Bobine behavior for text, tables, images, and rendering.
3. Prefer a single opened `pdf_oxide` document. If the existing `api::Pdf` cannot share state with `PdfDocument`, measure and document the temporary dual-open cost before accepting it.
4. Keep generic `convert_pdf_source()` behavior useful for custom sources: they may provide structured spans, or rely on the empty defaults.
5. Treat parser errors as recoverable source-data failures where appropriate; log/report them and allow the existing PDF/OCR path to continue.

**Acceptance:** the production adapter can query a tagged tree and list embedded files without adding another PDF crate.

### Phase 3 — tagged `/Formula` and MathML extraction

**Goal:** create reliable `MathSpan` values from the `pdf_oxide` parsed tree.

1. Traverse `StructTreeRoot::root_elements` in document order and find `StructType::Formula`, including custom roles mapped to Formula by the parsed role map.
2. Recursively inspect `StructChild::StructElem` and retain page association. Use marked-content references and MCID scope to associate a formula with the correct page content.
3. Convert MathML children to XML, then call the existing `mml::mml_to_tex`.
4. Keep `pdf_oxide` responsible for PDF objects, xrefs, references, and stream decoding. In 0.3.78, use `catalog()`, `resolve_object()`, and `load_object()` for a bounded, formula-focused raw lookup because the parsed `StructElem` projection drops the original object reference and skips `/A` and `/Alt`.
5. Align that lookup with the parsed projection before consuming raw metadata: use the already-parsed `tree.role_map` for the same `/RoleMap` decisions, `/K` document order, page inheritance/resolution, cycle policy, and traversal/time bounds. Verify formula counts/order and page/MCID signatures against the parsed tree; if the projection and raw walk disagree or hit a bound, do not attach raw `/AF` data for the affected scope and report a warning. The locked API does not expose the parser's internal page map or visited set, so this alignment check is a Bobine responsibility.
6. Follow `/AF` only after a raw node is confirmed to be a `/Formula` (including a custom role mapped to Formula). Read `/A`, `/Alt`, and `/NS` from that same raw dictionary. Preserve `/A` and `/NS` as normalized JSON on the Rust `MathSpan`; decode `/Alt` as a PDF text string, not as LaTeX. The public `pdf_oxide::fonts::font_dict::pdfdoc_encoding_lookup()` supports PDFDocEncoding alongside standard UTF-16 BOM handling. Apply the separate `max_formula_attachment_bytes` limit (1 MiB by default) to each associated MathML file after extraction.
7. Respect encrypted-document state for both raw lookups and attachments. Do not bypass `is_authenticated()` or turn an authentication failure into empty bytes; report the source as unavailable and let normal PDF/OCR conversion continue.
8. Reuse `extract_embedded_files()` when an `/AF` filespec is also registered in `/Names /EmbeddedFiles`; match only a unique filename/spec candidate, and retain the raw filespec's `/AFRelationship` and embedded stream `/Subtype` as metadata. `extract_embedded_files()` loses filespec identity and metadata, so ambiguous filename matches must be skipped. If an `/AF` filespec is absent from the name tree, leave its bytes unresolved rather than decoding the raw stream in Bobine.
9. Apply precedence: structure-element MathML, then successfully matched formula-associated MathML, then description-only Alt. Never convert Alt prose into a formula.
10. Set `bbox: None` when geometry cannot be established. Use MCID-to-text positioning only when the extracted content mapping is unambiguous.
**Acceptance:**

- structure-only and name-tree-registered associated-file fixtures produce the expected LaTeX and origins;
- when both are present, structure MathML wins;
- Alt-only remains a description and is not emitted as LaTeX;
- malformed MathML is reported without panics or unbounded traversal;
- structured formulas do not require TexTeller.

### Phase 4 — embedded document attachments

**Goal:** surface and route supported catalog attachments through `pdf_oxide`.

1. Implement `PdfSource::embedded_files()` using `PdfDocument::extract_embedded_files()`.
2. Preserve the filenames and decoded bytes returned by pdf_oxide. Do not claim that this API supplies MIME type, `/AFRelationship`, or filespec identity; enrich metadata only from a matched raw formula filespec.
3. Check each returned decoded size against `max_embedded_bytes` (16 MiB by default) before parsing document-level embedded sources. This is a downstream parsing cap: in 0.3.78, `extract_embedded_files()` decodes first and offers no per-call decompression limit. Do not claim the Bobine cap bounds peak decompression memory.
4. Route `.mml` / MathML XML through `mml_to_tex`.
5. Keep `.tex` as extracted bytes in the MVP; do not execute it or add a full TeX-to-Markdown converter.
6. Treat embedded JATS as a whole-document source only when explicitly selected and validated. Do not splice a complete JATS Markdown document into PDF pages, which can duplicate the document. If per-formula extraction from JATS is later needed, add an explicit formula-to-page mapping design.
7. Ignore unknown attachment types without failing PDF conversion.

**Acceptance:** embedded MathML is converted, oversized payloads are rejected before downstream parsing, and unrelated or unknown attachments do not alter Markdown output.

### Phase 5 — source-first routing and mixed pages

**Goal:** splice structured math while preserving OCR for uncovered formulas.

1. Collect structured spans before routing shortcuts in `route_page_inner()`, so the structured path works when ONNX is disabled or routing mode is Never.
2. Build the existing fast text-layer Markdown and place each structured span using reliable bbox/MCID/text association.
3. Filter only OCR formula boxes that overlap a structured span; increment `suppressed_ocr_spans`.
4. Continue existing OCR routing for untagged formula boxes on the same page.
5. Reuse the existing replacement/splicing conventions. Add display fences and equation tags only at Markdown emission time.
6. Preserve the current image interleaving and page-order behavior.

**Acceptance:** tagged-only pages make zero ONNX calls; mixed tagged/untagged pages retain both source math and OCR output; existing untagged-PDF goldens remain unchanged.

### Phase 6 — validation, reporting, and confidence

**Goal:** make source quality visible and prevent malformed output.

1. Reuse `sanitize_latex()` for repairs, recording whether it changed or rejected a formula.
2. Preserve MathML-to-LaTeX conversion behavior and its existing differential fixture gate.
3. Keep KaTeX validation behind an optional feature and default it off in the core crate.
4. Record source counts, suppressed OCR spans, validation results, and warnings in `MathSourceReport`.
5. Keep the existing render-health gate as its pdf_oxide-version-specific corruption workaround; it is separate from formula validation.
6. Expose the report and any supported configuration through Rust and Python APIs and document the stable output contract.

**Acceptance:** malformed output is reported, valid structured formulas retain expected output, and validation can be disabled without changing existing behavior.

### Phase 7 — benchmarks and follow-up export

**Goal:** measure the structured path separately from OCR quality.

1. Benchmark tagged synthetic PDFs with exact MathML ground truth.
2. Compare extracted formula streams with the same formulas passed through `convert_jats`.
3. Report formula match, elapsed time per page, and ONNX calls.
4. Keep CID-renderer-dependent OCR measurements separate until the renderer issue is resolved or a benchmark-only renderer is available.
5. Defer HTML/MathML export; Markdown remains Bobine's primary output contract.

---

## 6. Implementation sketches

### 6.1 Traverse the parsed structure tree

The implementation should consume `StructTreeRoot` and its parsed children. The exact code should follow the locked `pdf_oxide` public types:

```rust
let Some(tree) = document.structure_tree_with_budget(parse_budget)? else {
    return Ok(Vec::new());
};

for root in &tree.root_elements {
    collect_formula_spans(root, &tree.role_map, &mut spans)?;
}
```

The walker should match `StructType::Formula`, recurse through nested `StructChild::StructElem`, and record page/MCID references without inventing geometry. Read missing namespace or association data through `PdfDocument` object accessors only when needed.

### 6.2 Convert a MathML fragment

```rust
fn math_span_from_mathml(
    xml: &str,
    page: usize,
    bbox: Option<Rect>,
    origin: MathOrigin,
) -> Result<MathSpan> {
    let parsed = roxmltree::Document::parse(xml)
        .map_err(|e| BobineError::Other(format!("MathML parse: {e}")))?;
    let node = parsed.descendants()
        .find(|node| node.has_tag_name("math"))
        .ok_or_else(|| BobineError::Other("MathML has no <math> element".into()))?;

    let mut stats = MmlStats::default();
    let latex = crate::mml::mml_to_tex(node, &mut stats);
    // Apply the existing sanitizer and retain repair/failure details.
    // Markdown fences are added by the converter.
    // ...
}
```

This is a sketch, not an assumption that `pdf_oxide` currently emits serialized MathML XML. The implementation must establish the structure-element-to-XML mapping from the available parsed/raw `pdf_oxide` data first.

### 6.3 Route structured and OCR spans together

```rust
let structured = if self.config.math.prefer_structured_math {
    pdf.structured_math(index)?
} else {
    Vec::new()
};

// Merge structured spans into fast Markdown.
// Remove only OCR boxes covered by a positioned structured span.
// Pass remaining boxes through the existing routing mode.
```

A page with one tagged formula and one untagged formula must keep both outputs.

---

## 7. Validation plan

### Unit and fixture checks

- `PdfSource` default methods return empty results.
- `MathOrigin` serialization and `MathSourceReport` accumulation.
- Structure tree traversal identifies standard and role-mapped Formula elements.
- MathML namespace, `/ActualText`, `/Alt`, and `/AF` are handled according to data actually exposed by `pdf_oxide`.
- Embedded file names and decoded bytes use `extract_embedded_files()`.
- Budget and malformed-object behavior is bounded or reported.
- Only overlapping OCR boxes are suppressed.
- Mixed tagged plus OCR page preserves both formulas.
- Existing JATS and MathML differential fixtures remain unchanged.

### Golden and compatibility checks

- Existing untagged PDF goldens remain byte-identical.
- New synthetic tagged fixtures cover structure-only, associated-file-only, both, Alt-only, embedded MathML, and malformed data.
- Old serialized `ConverterConfig` documents still deserialize.
- Python configuration and report surfaces match the Rust API.

### Performance and CI checks

- Tagged fixtures make zero ONNX calls.
- Measure structure parsing and attachment extraction cost on representative PDFs.
- Build the normal configuration and any optional validation feature.
- Keep fixtures out of the published crate package.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `pdf_oxide`'s high-level structure model omits MathML namespace or attributes | Audit the locked version first; use its public object accessors or add/request a focused `pdf_oxide` API |
| `/AF` association is not returned by document attachment enumeration | Keep formula-level `/AF` distinct; resolve via the same `PdfDocument` object model |
| `extract_embedded_files()` lossily converts UTF-16 `/UF` filenames | Reconcile returned names against the same name-tree filespec objects; omit lossy-name collisions |
| attachment API decodes before Bobine can enforce a size limit | Verify backend stream limits; add/request bounded extraction before claiming a decompression cap |
| no reliable formula bbox | Use page/MCID/text association only when unambiguous; leave bbox absent otherwise |
| tagged and untagged formulas coexist | Suppress OCR locally per overlapping span, never per whole page |
| MathML conversion quirks | Reuse `mml.rs` and preserve its fixture behavior |
| PDF parsing adds memory or latency | Prefer one `PdfDocument` instance; measure any adapter that opens the same file twice |
| structured data is malformed or cyclic | Use `pdf_oxide`'s bounded parser where appropriate, add Bobine traversal limits, and fall back to existing conversion |
| Python and Rust APIs drift | Update bindings, stubs, config tests, and docs together |

---

## 9. Success criteria

1. Tagged PDF formulas produce LaTeX through the `pdf_oxide` structure-tree path.
2. Formula-associated MathML follows structure-MathML precedence when the backend exposes the association.
3. Catalog embedded files are extracted through `pdf_oxide::document::PdfDocument::extract_embedded_files()`.
4. Unsupported or missing semantic data falls back cleanly to Bobine's existing PDF/OCR path.
5. Mixed tagged and untagged formulas on a page are both preserved.
6. Existing untagged PDF output and MathML/JATS behavior remain stable.
7. Rust and Python callers can inspect math provenance.
8. Attachment handling enforces downstream decoded-size policy. **Peak decompression memory remains open** because the pinned backend decodes before Bobine receives the bytes.
9. A manual synthetic extraction benchmark reports formula count and latency. Structured accuracy and ONNX-call counts on a representative corpus remain open.

---

## 10. Open questions to close in Phase 0

1. Can Bobine use one `PdfDocument` instance for all current `PdfSource` operations, or does `api::Pdf` require a separate adapter?
2. **Resolved for 0.3.78:** the projection does not populate `/A` or `/Alt`, retain `/NS`, or retain each structure element's source object reference. Use the raw catalog/object APIs for formula-focused lookup, with fail-closed alignment to the parsed projection.
3. **Resolved for 0.3.78:** `/AF` can be followed from raw Formula dictionaries, but `extract_embedded_files()` supplies decoded bytes only when that filespec also appears in `/Names /EmbeddedFiles`; match by unique filename and leave unlisted or ambiguous files unresolved.
4. **Resolved for 0.3.78:** the public embedded-file API has no per-extraction size/decompression option. It decrypts and decodes before returning; Bobine can limit downstream parsing, not peak decompression memory. The separate public stream decoder's options do not provide document-level decryption.
5. Can the tagged tree's MCID references be mapped to reliable page positions through existing `pdf_oxide` text extraction?
6. Can the `pdf_oxide` writer create our synthetic tagged fixtures, or should the test fixtures be small, checked-in PDFs authored for the tests?

**Decision rule:** resolve these using the locked `pdf_oxide` APIs first. Any missing capability should be stated precisely before adding another PDF parser.

### Implementation progress

- Formula metadata now uses a bounded raw `/K` lookup against the same `PdfDocument`, then checks formula order, page mapping, MCIDs, and MCID scopes against the parsed projection before accepting `/AF`, `/A`, `/Alt`, or `/NS` data.
- Formula-associated bytes are reused from `extract_embedded_files()` only for a unique filename match. Those files are excluded from document-level MathML routing when structured math is enabled, preventing duplicate output.
- Raw `/A` and `/NS` are exposed as normalized JSON on `MathSpan`; `/Alt` is decoded as a description and remains separate from LaTeX.
- Synthetic PDF tests now exercise tagged Formula parsing, MathML-role conversion, page/MCID alignment, fail-closed alignment mismatch, structure-MathML precedence over `/AF`, associated-file-only MathML, Alt-only descriptions, raw `/A` and `/NS`, catalog attachment extraction, Unicode UTF-16 filename matching, unauthenticated encrypted documents, and a cyclic Formula kid. The formula `/AF` fixture is a small object-level test PDF; the PDF parser and stream decoder under test remain `pdf_oxide`.
- Converter coverage verifies that only the OCR candidate overlapping a tagged formula is suppressed and that a separate candidate remains available to produce an OCR formula. This uses the converter seam and does not load ONNX.
- A manual, ignored benchmark builds a synthetic 82,860-byte tagged PDF with 128 formulas and measures seven `structured_math()` runs. On the current Windows development machine it extracted all 128 formulas with a 75.006 ms median. Run it with `cargo test --lib tagged_math_extraction_benchmark -- --ignored --nocapture`. The benchmark excludes PDF open time, does not exercise attachments or ONNX, and is not a representative-corpus performance claim.
- Fixture coverage still needed for additional malformed structure-tree shapes and model-backed mixed tagged/OCR conversion. Accuracy and ONNX-call measurements on representative tagged PDFs also remain open.
- Release prep for 0.10.0 sets both manifests to `0.10.0`, adds the changelog entry, and makes the release workflow compile `math-export`.
- The pinned attachment API still decodes every name-tree attachment before Bobine can enforce parsing limits. The extraction code has a TODO for aggregate count/bytes and peak decompression memory; no decompression-memory bound is claimed.

---

## 11. References

- Bobine local PDF adapter: `src/pdf_source.rs`
- Bobine converter: `src/converter.rs`
- Bobine MathML converter: `src/mml.rs`
- Bobine JATS converter: `src/jats.rs`
- `pdf_oxide::structure`: <https://docs.rs/pdf_oxide/0.3.78/pdf_oxide/structure/>
- `PdfDocument` API: <https://docs.rs/pdf_oxide/0.3.78/pdf_oxide/document/struct.PdfDocument.html>
- PDF Association Best Practice Guide: Math in PDF: <https://pdfa.org/download-area/publications/BPG-Math-in-PDF.pdf>
- PDF Association, Deriving HTML from PDF: <https://pdfa.org/resource/deriving-html-from-pdf/>
- Bobine JATS/MathML proposal: `docs/proposal_jats_mml.md`
- Bobine renderer issue: `docs/cid_render_blocker.md`
