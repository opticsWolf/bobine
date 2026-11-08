# CID-renderer blocker: findings, options, status

**Date:** 2026-10-28; re-verified 2026-11-08. **bobine:** 0.7.1 (`dev`,
pushed) at original write; P2 POTATR routing since landed (`05dff78`,
still `dev`). **pdf_oxide:** 0.3.78 (still latest release, re-verified
via `cargo search` 2026-11-08; v0.3.79 still unreleased).
**Owner of next actions:** human reporter (draft in §5) + maintainer
waiting on synthetic reproducer; tracking in §4.

This is the consolidated record of the pdf_oxide CID-font render
corruption episode: what was found, what was ruled out, what is blocked,
and what remains. Detail lives in `docs/benchmarks.md` (P-D section),
`docs/handoff_gossamer_jats.md`, `docs/handoff_okfgraph_jats.md`, and
`docs/proposal_jats_mml.md`; this file is the index and the decision log.

---

## 1. Findings

### 1.1 Symptom

`RoutingMode::Always` output on the ikat corpus (10 PLOS ONE PDFs) is
glyph-garbled at high confidence: `tje` for `the`, `RESECRCJ CRTKCLE`
for `RESEARCH ARTICLE`, RapidOCR lines at 0.94–1.00 confidence,
TexTeller spans of body-text mbox garbage (`\[\mbox{or\;oulos…}\]`).
Rescoring with the fixed multi-line extractor changed nothing:
recall 31/196 displays, exact-match 0/31.

### 1.2 Root cause (verified, in order)

1. **The render itself is corrupt.** Rendering page 0 of
   `cs-bodvw-2024.pdf` at 150 dpi and viewing the PNG shows the glyph
   shift baked into pixels. OCR is innocent: it faithfully reads wrong
   pixels. (`target/probe/` artifacts were temp-only, since removed.)
2. **Counter-proof that the OCR stack is healthy.** `solitons.pdf`
   (arXiv-style, Type1-only fonts) renders and OCRs verbatim-correct at
   0.96–0.99 confidence, on CPU and on GPU (RTX 3090, ORT 1.28.1
   cuda13 + CUDA 13.3 + cuDNN 9.24). Same code, same models.
3. **Font structure of the failing files.** `cs-bodvw-2024.pdf`:
   Type0 fonts with **custom `/Encoding` CMap streams** (indirect
   refs, only 3× `/Identity-H`), **no `/CIDToGIDMap`**,
   embedded `/FontFile2` (TrueType subsets) ×3, `/ToUnicode` ×20.
   Text-layer extraction is correct (ToUnicode path fine); only the
   raster path shifts glyphs — consistent with rendering
   `CIDToGIDMap[code]` (identity fallback, no map) while ignoring the
   Encoding CMap, i.e. code-as-CID on subset fonts whose GIDs are
   arbitrary.
4. **2026-11-07: text layer corrupt too on PLOS-subset files.**
   Always+Potatr spot on `econ-tax-2019.pdf` (POTATR P2 validation)
   shows extraction-level mojibake independent of bobine code:
   pdf_oxide's own `to_markdown` emits `Tazatkon and economkc
   sorjkstkcatkon` and POTATR cell tokens read `Effectkve labor taz`.
   The §1.2(3) render-only finding was cs-bodvw-specific; on PLOS
   subset-font files the corruption reaches the char identities, so no
   text-backed path (fast, OCR-gated, or POTATR) can recover them.
4. **A second, unrelated engine bug found en route and fixed.**
   `RapidOCR::rec_input` computed `x/255 - 0.5`, missing the `/ 0.5`
   (PP-OCRv4 spec, the code comment, and the det path all say
   `(x/255-0.5)/0.5`). Fixed in `src/rapid_ocr.rs`, unit-test pinned.
   It did **not** cause the garbage (identical output before/after) —
   half-range inputs degrade quality but do not shift glyphs. Kept
   because it is spec-correct, not because it mattered here.

### 1.3 Upstream mapping (live-checked 2026-10-28)

- Repo: `github.com/yfedoseev/pdf_oxide`. Latest release **v0.3.78**
  (also tagged `Latest`); **v0.3.79 milestone exists but is unreleased**.
- Directly adjacent closed issues, all milestoned v0.3.79:
  - **#1500** — *"Type0 font's embedded /Encoding CMap stream is not
    used to map codes to CIDs, so glyphs render from
    CIDToGIDMap[code]"*. The mechanism matches §1.2(3) exactly.
    Our case is unconfirmed-fixed until re-tested on `main`/0.3.79.
  - #1626 (predefined CMap ≠ code-to-CID map), #1629 (missing `/DW`
    default), #1515 (non-embedded CIDFont substitution), #1374
    (`.notdef` GID 0 not rendered — mapping failures are *invisible*,
    consistent with our silent corruption).
- No open duplicate of our symptom found (searched `CIDFont render`,
  `wrong glyph Type0` across 373 issues).

### 1.4 Blast radius (narrower than the benchmark suggests)

| Population | Affected? | Why |
|---|---|---|
| Scanned PDFs (OCR path's primary users) | **No** | Page images have no fonts/Encodings/CMaps |
| Born-digital, Type1/TrueType fonts | **No** | solitons proof |
| Born-digital, custom-Encoding CID subset fonts (PLOS-style) via PDF-OCR route | **Yes** | this blocker |
| Same papers via JATS route | **No** | no rendering involved; 692/692 + 10/10 green |
| Fast path (text layer) on affected files | **No** | ToUnicode path is correct |

The ikat corpus happens to be the worst case (all PLOS ONE). The
gossamer `prefer_jats` design (§5 handoff) already routes the scholarly
population around the bug: it blocks a fallback path, not the
recommended one.

### 1.5 Issue-filing requirements (pdf_oxide repo rules)

Full policy pulled from `ISSUE_TEMPLATE/bug_report.yml`,
`ISSUE_TEMPLATE/config.yml`, and `CONTRIBUTING.md` (all live-fetched):

- Must use the **Bug Report form** (`blank_issues_enabled: false`;
  unfilled reports auto-close). Auto-labels `bug`+`triage`.
- Required: description, exact reproduction steps, expected/actual,
  environment (OS, `rustc`, crate version), and four checkboxes —
  duplicate-search done, reliably reproducible, **no copyrighted /
  third-party PDF attached**, own-words authorship.
- **Our PLOS PDFs can never be attached.** Compliant path: describe
  structure (§1.2(3)) or build a minimal synthetic Type0 PDF in code
  (their `tests/` pattern; `fixture-hygiene` CI rejects third-party
  fixtures; `TESTS fixtures/TRACKED_PDFS.txt` governs exceptions).
- **Agents may not file.** Issues/PRs/descriptions must be human-written;
  AI assistance disclosed via `Assisted-by:` trailer. A draft can be
  prepared here; the human rewrites and submits.
- Rate limits are automated (<5 min between issues auto-closes).
- PRs later need an approved + assigned issue, one open PR per
  contributor, a failing-before/passing-after regression test, and a
  corpus sweep on the contributor's own corpus vs *two* baselines.

Filing readiness: latest-version ✓, duplicate search ✓, environment
facts ✓ (Win11 b26200, rustc 1.99.0, pdf_oxide 0.3.78), structure
description partial ✓, **synthetic reproducer ✗, human-authored report
✗, main-vs-.78 verification ✗**.

### 1.6 Option A outcome (2026-11-08): NOT fixed on `main`

Shallow clone @ `08accb2` (2026-10-30, post-0.3.78), built
`pdf_oxide_cli` (CLI self-reports 0.3.78 — version unbumped), rendered
`cs-bodvw-2024.pdf` p0 @150dpi via `pdf-oxide render --dpi 150 -p 1`:
byte-identical to the 0.3.78 render (SHA-256 `e8b473ce…` both files,
`ImageChops.difference` sum 0). Same glyph-shifted pixels
(`RESECRCJ CRTKCLE`, `deer learnkng`, `Cbst ract`). The #1500 mechanism
(Type0 embedded `/Encoding` CMap ignored at render) is still present
on `main`; no fix to wait for. The upstream issue for the residual
variant is now unblocked on content — remaining: synthetic reproducer
+ human authorship (draft below).

---

## 2. Options analysis

### Option A — verify `main` fixes our case (investigative, ~30 min compute)

~~Shallow-clone pdf_oxide, render one PLOS page on `main`, diff against
0.3.78.~~ **Executed 2026-11-08, result in §1.6: NOT fixed** (byte-identical
garbage). Shallow-clone pdf_oxide, render one PLOS page on `main`, diff
against 0.3.78. Highest value per effort: a fix dissolves the problem
(plan becomes "wait + pin"); a non-fix *is* the upstream issue content
(residual variant of #1500). **Prerequisite for every version-based
plan. Recommended first.**

### Option B — dev-only git dependency on pdf_oxide `main`

Unblocks internal P-D runs early; zero bobine code. **Cannot ship:**
`cargo publish` refuses git dependencies, and `[patch]` sections are
*stripped* on publish — a forgotten revert silently ships different
behavior than dev tested. Needs a loud CI guard; must revert before any
release. Internal-numbers only, never users.

### Option C — version-scoped render-health gate (the shippable workaround)

Per page, on born-digital input with substantial text layer: after
full-structure OCR, compute OCR↔text-layer agreement (word-Jaccard on
alphanumeric tokens). Below threshold **and only while built against
pdf_oxide < 0.3.79** (explicit flag flipping on the dep bump = automatic
expiry): discard OCR, emit fast-path markdown, log + metric. Zero extra
cost (OCR already ran).

Math handling is suppression-first: TexTeller spans on gate-fired pages
are dropped (confident false equations are worse than none); *optionally*
salvage inline-only spans from text-layer runs (unicode + baseline-shift
scripts → fused `$…$`; displays never attempted — text-layer glyphs
cannot yield `\frac`/`\sqrt`/matrix structure, and scans have no layer
at all). Gate fires, suppressed-span counts, and salvaged-span counts
must be loud (bench outputs + report) so the harness can never silently
score fallback output as OCR output — otherwise the gate contaminates
the very benchmark it is meant to protect. Reverse-case risk (broken
ToUnicode + good render) is contained by the version scoping. Arguably
permanent value as a render-regression guard beyond 0.3.79.

### Dismissed

- **Font-sniffing per-page fallback:** no public font-introspection API
  on `Pdf` (full surface grepped); driving `fonts::font_dict`
  internals = a second PDF parser, version-coupled. Dead.
- **PDF rewrite** (remap content codes→CIDs, force Identity-H, rewrite
  ToUnicode): sound against the #1500 mechanism, but hundreds of lines
  of content-stream surgery (TJ kerns, multibyte codes, object streams,
  encryption) as *temporary* code. Wrong risk profile.
- **External renderer** (poppler etc.): system dependency,
  platform-specific, breaks hermetic wheels. No.

### Verdicts

- P-D numbers: **no good option exists** — correct pixels are the
  prerequisite and only the fixed renderer makes them. The table stays
  withheld; manufacturing numbers via fallback would be measuring the
  wrong path under the right label. Refusing that is the decision
  (logged, kept).
- User-facing quality: Option C is a genuine improvement
  (garbage → correct text + honestly-suppressed math), shippable and
  reversible. Optional, not required.
- Default plan if nothing else is chosen: wait + pin, with the JATS
  route covering scholarly papers meanwhile.

---

## 3. Current status

**Done (all on `dev`, pushed):**

- P-D harness `tests/test_bench_ocr.rs` (Always, order alignment,
  normalized exact match, `report.md`/`mismatches.md`, rescore mode);
  accuracy table withheld with analysis in `docs/benchmarks.md`.
- `rec_input` normalization fix (spec-correct, pinned).
- Deferred media trio (EmbedBase64, side-by-side captions,
  preserve_layout) + Python mirrors; P-E parity (README Structured
  sources, module maps, quickref, CHANGELOG 0.7.1).
- GPU path validated (RTX 3090, ORT cuda13 + cuDNN 9.24).
- Both downstream handoffs: `docs/handoff_gossamer_jats.md` (§5),
  `docs/handoff_okfgraph_jats.md` (§6).
- §6.5 interim: 10 ikat GT markdowns in `kb.db` as
  `ikat-ground-truth/*` (~1,425 chunks), search-verified.

**Blocked (named unblock conditions):**

- P-D accuracy table → pdf_oxide 0.3.79 release (or Option A
  confirming `main`, then a release pin).
- okfgraph `formula_concepts` for `kind='pdf'` → same renderer fix
  (mining formulas from OCR-mojibake would enshrine garbage; JATS
  formula concepts unaffected).
- Upstream issue → ~~synthetic reproducer + `main` verification +~~ human
  authorship (§1.5). Synthetic reproducer BUILT + verified 2026-11-08
  (matrix in §5.1): `cid-repro-type0.pdf` (custom `/Encoding`) +
  `cid-repro-A.pdf` (Identity-H subset control); pdf_oxide 0.3.78 AND
  `main`@`08accb2` render blank where PyMuPDF renders/extracts ABCDEF.
- POTATR default-on (Auto triple-gate leg 3, "trustworthy renders") →
  same renderer fix; P2 opt-in validated on clean renders only
  (`05dff78`; econ-tax spot: correct 13-col structure, CID-garbage
  text — garbage-in/garbage-out at the text layer).

**Corpus note (2026-11-07):** `ikat-corpus/` is untracked (gitignored),
so corpus pages can't be locked in-tree. `chem-olive-2024.pdf` changed
since the parity fixtures (now 33 pages; the `potatr_gate_chem_p5`
fixture shows PLOS ONE 6/11) — page-index references to corpus PDFs
rot; cite content, not indices.

**Open (no owner yet):** Option A execution; Option C build decision;
gossamer §5 + okfgraph §6.1–6.4 implementation (external repos).

---

## 4. Decision log

1. Withhold P-D numbers rather than publish renderer-measuring ones
   (2026-10-28; rationale §2-verdicts).
2. Keep the `rec_input` fix despite it not explaining the symptom
   (spec-correctness; zero behavior risk to healthy paths).
3. No PDF-rewrite or external-renderer workarounds (risk/coupling
   disproportionate to a temporary shim).
4. JATS-first as the strategic answer for scholarly PDFs (already
   specified to gossamer; unaffected by this blocker).
5. P2 POTATR routing validated on clean renders only; CID-corrupt text
   layers fail the claim-quality gate or yield garbage-in cells — no
   new fallback built, triple-gate leg 3 holds (2026-11-07).
6. Docs refreshed 2026-11-08 ahead of Option A execution (0.3.78 still
   latest, 0.3.79 unreleased; text-layer corruption finding; corpus
   drift note).
7. Option A executed 2026-11-08: `main` renders byte-identical garbage
   (SHA-256 match) — no fix to wait for; upstream issue unblocked on
   content, draft prepared for human authorship.
8. Synthetic reproducer built + verified 2026-11-08 (user-built, Arial
   source): `cid-repro-type0.pdf` + `cid-repro-A.pdf` (paths+hashes in
   §5.1). pdf_oxide 0.3.78 AND `main` render blank / extract FFFD
   where PyMuPDF renders + extracts ABCDEF — proves a valid-file
   render failure, not a malformed-file artifact. Remaining for filing:
   human rewrite/submit of the §5 draft with reproducer attached.

## 5. Draft upstream report (human to rewrite + submit — agents may not file)

### 5.1 Synthetic reproducer (built + verified 2026-11-08)

Two minimal single-page Type0 files (user-built with fontTools from
`C:/Windows/Fonts/arial.ttf`; subset keeps A–F + .notdef, upm 2048):

- `D:/User/Documents/Rust/cid-repro-type0.pdf` (23,083 B, SHA-256
  `e669747c f97444d9 68dd7d10 84949790 673649e4 251ac62b 0d2e3321b2b84dd2ad`):
  custom `/Encoding` CMap stream (`<41>`–`<46>` → CID 1–6),
  `CIDFontType2` subset, NO `/CIDToGIDMap` (identity default),
  embedded `/FontFile2`, `/ToUnicode` covering post-Encoding CIDs
  (`<0001>`–`<0006>` → U+0041–U+0046) AND raw codes
  (`<0041>`–`<0046>` → U+0041–U+0046), content `(ABCDEF)`.
  PLOS-faithful shape (custom Encoding + subset + no CIDToGIDMap).
- `D:/User/Documents/Rust/cid-repro-A.pdf` (23,106 B, SHA-256
  `e30de26a ed72f067 17921a4b 5c0bf5b6 dfd06dbcf 2825979e a8042efb9 69d0e0`):
  same subset font but `/Encoding /Identity-H`, 2-byte codes
  `<0001>`–`<0006>`. Subset control (no custom Encoding involved).

Behavior matrix (all verified 2026-11-08):

| file | PyMuPDF render | PyMuPDF extract | pdf_oxide 0.3.78 render | pdf_oxide `main`@`08accb2` render | pdf_oxide extract |
|---|---|---|---|---|---|
| `cid-repro-type0.pdf` | ABCDEF ✓ | `ABCDEF` ✓ | blank ✗ | blank ✗ (byte-identical to 0.3.78) | 3×U+FFFD ✗ |
| `cid-repro-A.pdf` | ABCDEF ✓ | `ABCDEF` ✓ | blank ✗ | (same build, same path) | — |

Reference renders kept at `C:/Users/Main/AppData/Local/Temp/repro_A_pym.png`
(PyMuPDF ABCDEF) vs blank pdf_oxide PNGs (ink-sum = pure white).

What the matrix proves: the file is VALID (independent reader renders
+ extracts correctly via Encoding CMap → CID → GID-identity → subset
outlines). pdf_oxide fails the SAME bytes on both render and extract
paths, on both 0.3.78 and `main`. The Identity-H control (`A`) shows
the subset font never draws in pdf_oxide even on the trivial path, so
the report should frame the primary file as "valid Type0 subset file
renders blank" with the custom-Encoding structure as the PLOS link
(codes 0x41–0x46 resolved as CIDs 65–70 → GIDs out of the 7-glyph
subset range → skipped), not over-claim a single internal mechanism.
Dead ends excluded from the report: full-font Identity-H probes drew
`$%&'()` in BOTH readers (CID-as-codepoint→cmap consensus on CIDs
36–41 — no divergence, file-design artifact, files deleted); ToUnicode
named `Adobe-Identity-UCS` is silently replaced by readers' builtin
identity (use a custom CMapName).

Title: `Type0 render ignores embedded /Encoding CMap: glyphs shift on
subset fonts without /CIDToGIDMap`

Body skeleton (own words required, `Assisted-by:` trailer):

1. Repro: two minimal synthetic Type0 files (attached, no third-party
   PDF per repo rules — public-domain Arial-subset construction):
   `cid-repro-type0.pdf` (custom `/Encoding` CMap stream, indirect
   refs, no `/CIDToGIDMap`, embedded `/FontFile2` subset, content
   `(ABCDEF)`) and `cid-repro-A.pdf` (same subset, `/Identity-H`).
   `pdf-oxide render --dpi 150 -p 1` on either → blank page; PyMuPDF
   renders AND extracts `ABCDEF` from the same bytes. Same blank on
   0.3.78 and `main` @ `08accb2` (byte-identical PNGs). On real-world
   PLOS ONE pages (Type0 subsets, custom Encoding, no CIDToGIDMap)
   the same defect shifts glyphs instead of blanking (`RESEARCH
   ARTICLE` → `RESECRCJ CRTKCLE`) — codes resolved as CIDs land on
   arbitrary in-range subset GIDs there, out-of-range here.
2. Structure: `Type0` + custom `/Encoding` CMap stream + no
   `/CIDToGIDMap` + `/FontFile2` subset; ToUnicode present so text
   extraction is correct — only the raster path shifts, consistent
   with `CIDToGIDMap[code]` identity fallback ignoring the Encoding
   CMap (code-as-CID on arbitrary subset GIDs). Adjacent to #1500;
   persists after it.
3. Env: Windows 11 b26200, rustc 1.99.0, pdf_oxide `main` @ `08accb2`
   (CLI reports 0.3.78) and crates.io 0.3.78 — identical bytes.
