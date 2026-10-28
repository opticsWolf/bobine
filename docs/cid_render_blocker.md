# CID-renderer blocker: findings, options, status

**Date:** 2026-10-28. **bobine:** 0.7.1 (`dev`, pushed). **pdf_oxide:**
0.3.78 (latest release, verified via `cargo search` today; v0.3.79
unreleased). **Owner of next actions:** see §6.

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

---

## 2. Options analysis

### Option A — verify `main` fixes our case (investigative, ~30 min compute)

Shallow-clone pdf_oxide, render one PLOS page on `main`, diff against
0.3.78. Highest value per effort: a fix dissolves the problem (plan
becomes "wait + pin"); a non-fix *is* the upstream issue content
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
- Upstream issue → synthetic reproducer + `main` verification + human
  authorship (§1.5).

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
