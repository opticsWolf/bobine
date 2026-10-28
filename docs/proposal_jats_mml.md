# Proposal: JATS/MathML formula ingestion — `mml_to_tex` in Rust

Status: P-A + P-B + P-C implemented in 0.7.0 (P-D harness implemented in
0.7.1 with the accuracy table withheld — pdf_oxide renderer blocker, see
benchmarks.md; P-E okfgraph/gossamer follow-ups open)
Scope: bobine Rust core (`mml.rs` new, `converter.rs`, `config.rs`, `py_bindings.rs`) + test harness; continuation phases in okfgraph (`components/ingest.py`, `components/converters.py`) and gossamer (`document.py`, `open_access.py`)
Related: [architecture.md](architecture.md) · [benchmarks.md](benchmarks.md) · [proposal_media_tables.md](proposal_media_tables.md) (house format)
Provenance: the converter and the ground-truth corpus come from the ikat M7.8 work (`ikat/examples/auto-proof/mml.py`, 488 lines, stdlib-only; 10-paper corpus, 511 display formulas with publisher-MathML-derived LaTeX verified by TeX compilation and a width audit to 0 overfulls).

---

## 1. Problem statement

### Formulas today

| Path | Behaviour | Defect |
|---|---|---|
| Born-digital text layer (`pdf_oxide`) | math glyphs arrive as loose Unicode runs (`ρ`, `ζ`, `∗`) with no structure | fractions lose their bar, sub/superscripts flatten into the line; nothing can line-break |
| `math_boxes_from_chars` + TexTeller OCR (`Surgical`/`Always`) | formula crops rendered at 300 dpi → ViT encoder → RoBERTa decoder → LaTeX | OCR is probabilistic: unbalanced `\left`/`\right`, double superscripts, shredded commands, wrong symbols. Measured on the ikat corpus, PDF-side extraction flattened 511 formulas to glyph soup (993 glyphs became `?`, 70 overfulls to 494 pt) before the structured pipeline replaced it |
| Downstream consumers (okfgraph `kind='pdf'` ingest, any LLM chunking) | receive whatever the above produced | garbled formulas embed as noise; semantic search over `?`-strings is meaningless |

### The structured-source gap

Publishers ship the *same* papers with formulas as **structured MathML** (PLOS/PMC/EuropePMC JATS) or **arXiv LaTeX source**. The ikat M7.8 work proved the delta on a real corpus: reconstructing LaTeX from publisher MathML took math-hilfer-2024 from 70 horizontal overfulls to **0**, with every one of its 100 display spans classifying consistently against TeX-measured widths. That converter (`mml.py`) is stdlib-only (ElementTree + `re`) and is the basis for this proposal.

**Design principle carried over from ikat:** when the caller can obtain structured math (JATS, arXiv LaTeX), prefer it; use the PDF/OCR path only for what the structured source does not carry, and as the fallback when no structured source exists. The PDF stays the universal *fallback*, never the preferred formula source.

---

## 2. Design proposals

### P-A. `src/mml.rs` — Rust port of `mml_to_tex` (the core)

Port `ikat/examples/auto-proof/mml.py` (488 lines) into a self-contained
`mml.rs` with no new dependencies (XML parsing via the existing ecosystem;
`roxmltree` is the small candidate if no XML crate is already in the tree —
bobine currently has none, so this is the only new dependency of P-A).

**Public surface:**

```rust
/// One `<mml:math>` element (or a bare formula element) -> bare LaTeX
/// (no `$` wrapping; the caller owns math mode). Closes stray stretchy
/// fences: a lone stretchy `{` gets its `\right.` so the span compiles.
pub fn mml_to_tex(node: &Node, stats: &mut MmlStats) -> String;

/// Fuses JATS-split math runs (publisher markup splits one formula
/// across `<inline-formula>` runs / styled text runs).
pub fn merge_math(par: &str) -> String;      // port of merge_math

/// Unicode math letters/ops -> ASCII or TeX (stray text inside math).
pub fn mathify(text: &str, stats: &mut MmlStats) -> String;

/// Longest-KNOWN-prefix command splitter (see 2.A.3).
fn split_cmds(tex: &str) -> Cow<str>;
```

#### 2.A.1 Element dispatch table (port of `_tex`, `mml.py:200`)

| MathML element | LaTeX emission | Notes from the Python original |
|---|---|---|
| `mi` | per-char: `GREEK[ch]` map, else ASCII pass, else `mathify` | honours `mathvariant`; NFKC-decomposition fallback (`mml.py:141`) |
| `mn` | digits via `mathify` | — |
| `mo` | `{`/`}` → **always** `\left\{`/`\right\}` (cases braces; JATS never marks stretchy); `\|` → plain `\|` (never scales: kets `\|γ⟩` have no partner); `()[]` with `stretchy="true"` → `\left..\right`; single-char `OPS[txt]`; else `mathify` | the scale/no-scale split is measured — cross-group `\left..\right` is a fatal TeX group error |
| `mtext` | `\text{…}` with text escaping | — |
| `mspace` | `width` in em → `\quad` (≥1em) / `\;` (≥0.2em) / `\,` | — |
| `mrow`, `mstyle`, `mpadded`, `mphantom`, `semantics` | transparent: recurse over kids | `semantics` strips `<annotation>` and keeps only the first non-annotation kid |
| `mfrac` | `\frac{num}{den}`; <2 kids → join kids + `mml_short_frac` stat | — |
| `msqrt` | `\sqrt{…}` over joined kids | — |
| `mroot` | `\sqrt[deg]{rad}` (kid 2 = degree); <2 kids → `\sqrt{…}` | — |
| `msub`/`msup`/`msubsup` | base + `_{sub}`/`^{sup}`; **empty base → bare prescript** (`^{H}` floats left); multi-letter bare base braced `{base}` (single char, single `\\cmd`, or `\left(`/`(`/`[` stay bare); same-level nesting (`x^a^b`-shaped, prescript-as-msup-on-msup) separated by `{}` via `trailing_scripts` — TeX forbids double scripts at one level | `mml.py:231–278` |
| `munder`/`mover`/`munderover` | base in `LIMIT_OPS` → stacked `_{under}`/`^{over}` (limits stay under/over); single-char `over` in `ACCENTS` → `\hat{base}` etc.; else `\underset{under}{base}` / `\overset{over}{base}` | the `\underset`/`\overset` form measured exactly on composites |
| `mtable` | `\begin{matrix} a & b \\ c \end{matrix}` (rows via `mtr`/`mlabeledtr`, cells via `mtd`, `&`-joined) | increment `mml_matrix` stat |
| `mfenced` | `open` + sep-joined kids + `close` (attributes with defaults `(`/`)`/`,`) | — |
| `annotation` | `""` | — |
| unknown | kids joined transparently + tag recorded in `mml_passthrough` stat (capped list, 12) | never a hard error — conversion is best-effort by contract |

**Data tables to port verbatim** (`mml.py:41–107`): `GREEK` (47 entries incl.
`\hbar`, `\wp`, blackboard/calligraphic letters), `OPS` (the `<mo>` map),
`LIMIT_OPS`, `ACCENTS`, and the tex-env constants. These are pure data —
port as `static` arrays/`phf`-style matches, unit-test a handful of
representatives.

#### 2.A.2 Structural repairs (port) vs corpus hacks (do not port)

Port — these are generic MathML/publisher failure modes:

* `_split_cmds` (longest-**known**-prefix command splitting): TeX reads the
  longest alpha run after `\`, so `\infty}` is one command, but publisher
  output contains undefined `\inL` which must split after the longest known
  prefix (`\in` + `L`). A regex alternation cannot do this (it backtracks
  and shreds `\infty` → `\inf ty`); port the prefix-scan against
  `_known_cmds` (`mml.py:331`). `_known_cmds` itself ports as a
  `static` set (LaTeX base + amsmath + amssymb command inventory).
* Unclosed-fence fixup at the root: `\left` count − `\right` count > 0 →
  append `\right.` ×n, count in `mml_unclosed_fence` stat.
* `\Vert` norms stay plain (positional pairing cannot respect TeX grouping;
  single bars never scale).
* Same-level script separation via `trailing_scripts` (`mml.py:376`).

**Do not port:** the three publisher-misencoded glyph hacks
(`U+01BB`→`\zeta`, `U+049F`→`\beta`, `U+0277`→`k`, `mml.py:54`). Those are
PLOS-corpus-specific (roles proven per-paper from PDF renders + symbol
coexistence). In bobine they would corrupt unrelated publishers' documents.
They stay in the ikat exporter. The Rust `MmlStats` struct keeps the same
stat keys as the Python `stats` dict (`mml_unclosed_fence`,
`mml_short_frac`, `mml_matrix`, `mml_passthrough`) so the port can be
diffed against the Python original.

#### 2.A.3 Acceptance: differential test against the Python original

The Python `mml.py` is the reference implementation. Port the test corpus
mechanically:

* Extract every `<mml:math>` element from the 10-paper JATS set (511 display
  + inline formulas) into fixtures `<paper-id>-f<n>.mml` — **excluded from
  the crates.io package** per the existing `exclude = ["tests/fixtures/"]`
  rule (they ship in git only).
* Golden rule: `mml.rs::mml_to_tex` output **==** Python `mml_to_tex` output
  byte-for-byte on all 511 fixtures (after normalising the three
  corpus-specific glyph hacks out of both — the Rust port maps them to the
  passthrough branch; the Python side is run with a `SPECIAL = {}` shim).
* 511/511 identical = P-A done. Any divergence is either a port bug or an
  undocumented Python behaviour — fix the port, never "fix" the test.

### P-B. JATS input adapter — `convert_jats`

JATS is not page-based, so it mirrors `convert_office`, not `PdfSource`:

```rust
// converter.rs
impl HybridConverter {
    /// JATS XML (publisher full text) -> Markdown. Formulas: MathML ->
    /// LaTeX via mml.rs ($...$ inline, $$... \tag{N}$$ display).
    /// Figures: <graphic> -> staged PNGs (they are remote URLs in JATS —
    /// download or leave placeholders per config); tables: real JATS
    /// <table> -> GFM. Sections -> #/## headings. No ONNX engine needed.
    pub fn convert_jats(&mut self, path: &Path, work_dir: &Path) -> Result<JatsDocument>;
}
pub struct JatsDocument { pub md_path: PathBuf, pub n_formulas: usize, /* … */ }
```

Markdown shape **must mirror the ikat exporter's proven contract** (the
shape okfgraph lint and ikat's weaver already accept):

* inline math as single `$…$` spans (fused via `merge_math` before the
  weave guard counts `$`),
* display math as whole-paragraph `$$tex \tag{N}$$` fences (the tag rides
  inside, from the JATS `<label>`),
* the existing `plausible_display_latex` filter applies to the emitted
  display spans as a guard,
* hard-error policy mirrors bobine's fail-fast rules: unknown JATS elements
  pass through transparently (they are markup, not user error), but
  unbalanced `$` after fusion is a `BobineError`.

`ConverterConfig` gains no new routing mode — JATS vs PDF is a *source*
choice the caller makes (it must download the XML anyway). The existing
`RoutingMode` stays PDF-only.

### P-C. LaTeX sanitizer pass over TexTeller OCR output

Reuse `mml.rs`'s structural checks as a **post-OCR repair pass** on
TexTeller's decoded LaTeX (which exhibits the same failure classes):

1. `_split_cmds` normalisation (shredded/merged commands),
2. unclosed-fence fixup (`\left`/`\right` imbalance → `\right.`),
3. same-level script separation (`x^a^b` → `x^a{}^b`),
4. brace balance check (hard-fail the span back to raw text on imbalance —
   OCR output that cannot compile is worse than the text layer's loose
   glyphs).

Wired into `math_blocks` after `recognize_formula`, gated by a new
`ConverterConfig.text.sanitize_ocr: bool` (default **true** — bug-fix
positioning, same argument as ikat's typography guards).

### P-D. OCR benchmark harness vs the ikat ground truth

The ikat auto-proof corpus doubles as a labeled eval set for TexTeller:
10 papers, 511 display formulas, each with (a) publisher-MathML→LaTeX
ground truth (TeX-verified), (b) the original PDF.

* Harness lives in `tests/` (excluded from the crate), behind a feature or
  `#[ignore]` + env var like ikat's `dump_estimates`, since it needs the
  fixture PDFs + ONNX models.
* **Alignment**: bobine's OCR path localises formula regions itself
  (`math_boxes_from_chars`); align OCR'd spans to ground truth per paper by
  document order + section context (both sides carry the display sequence;
  JATS `<label>` tags anchor the ground-truth side).
* **Scoring**: normalized exact-match (whitespace + `\left`/`\right`
  equivalence classes) and, for mismatches, rendered-pixel diff (render
  both through a local TeX once, compare rasters) — the ikat session's
  `diff.py` pattern (`pymupdf`/`fitz`, `pix.samples`, `pix.n >= 3`).
* **Deliverable**: `docs/benchmarks.md` gains a TexTeller accuracy table
  (per publisher, born-digital vs scanned) — turning "80M training pairs"
  marketing into a measured number on real publisher PDFs.
  *Status 0.7.1: harness implemented as `tests/test_bench_ocr.rs`
  (Always mode, order alignment, normalized exact match, `report.md` +
  `mismatches.md`, rescore mode); accuracy table WITHHELD — the ikat
  corpus (PLOS ONE, Type0/CIDFontType2 fonts) renders glyph-corrupted
  under pdf_oxide 0.3.78, so all Always-mode OCR output inherits the
  corruption (details + counter-proof in `docs/benchmarks.md`). Also
  fixed en route: `rec_input` normalization now `(x/255-0.5)/0.5` per
  the PP-OCRv4 spec. Re-run once the upstream renderer is fixed.*

### P-E. Surface parity (bobine's own rule)

Same-commit updates for every surface bobine exposes:

* Rust: `mml.rs`, `convert_jats`, `JatsDocument` (P-A/P-B),
  `sanitize_ocr` (P-C),
* Python: `bobine.convert_jats`, `bobine.JatsDocument`,
  `bobine.ConverterConfig(..., sanitize_ocr=...)`,
* `python/bobine/__init__.pyi` stubs,
* README (module map + a "Structured sources" section),
* `docs/architecture.md` (module map + data flows),
* `docs/quickref.md`,
* `CHANGELOG.md` entry.

---

## 3. Version ladder

| Step | Scope | Version |
|---|---|---|
| 1 | P-C sanitizer (independent, immediately useful) | `0.6.1` |
| 2 | P-A `mml.rs` + differential fixtures | `0.7.0-alpha` (no release) |
| 3 | P-B `convert_jats` + P-E surface parity | `0.7.0` |
| 4 | P-D benchmark harness + `benchmarks.md` numbers | `0.7.1` |

Version increments follow the existing convention: `+0.0.1` minor features,
`+0.1.0` for the completed JATS/MML phase. `Cargo.toml`, `Cargo.lock`,
`pyproject.toml`, and `python/bobine/__init__.py` stay synchronized;
crates.io upload stays under 10 MB (fixtures excluded, already the rule).

---

## 4. Test plan

* **Unit** (in `src/mml.rs`, ship in the crate): one representative per
  dispatch-table row; `_split_cmds` cases (`\infty}` vs `\inL` vs
  `\inf ty`); prescript/same-level-script cases; fence fixup; `|` vs
  `\Vert` scaling; `mtable` nesting; stats counters.
* **Differential** (fixtures, git-only): 511/511 byte-parity vs Python
  `mml_to_tex` (P-A.3).
* **Golden snapshots**: `convert_jats` on one full paper per publisher
  family (PLOS, PMC) — md output frozen like `tests/golden/`.
* **TeX gate** (dev machine/CI, not crate tests): every display span from
  the golden JATS conversion must compile (pdflatex `\sbox` probe, the
  ikat `math_audit.py` pattern); every span's TeX-measured width within
  tolerance of the calibrated estimate.
* **Sanitizer**: TexTeller outputs with injected `\left`-imbalance,
  `x^a^b`, shredded commands — repaired or hard-failed as specified.
* **Benchmark** (`#[ignore]` + fixtures): P-D scoring; publish numbers in
  `docs/benchmarks.md`.

---

## 5. Continuation: gossamer (after bobine lands)

1. **JATS-first document route** (`gossamer/document.py`): in
   `DocumentExtractor.extract_document` (and `extract_document_structured`),
   add a structured-source step *before* the pdf_oxide path: paper URL/DOI →
   `OpenAccessLocator` → JATS (PLOS/PMC/EuropePMC full text) → fetch XML →
   delegate to bobine `convert_jats` (bobine is already the extraction
   engine family gossamer uses via oxide crates; the Python side only
   routes). Fall back to `pdf_oxide` when no structured source resolves.
2. **Config/surface**: `extract_document(source, prefer_structured=True)`
   (default on — the corpus evidence justifies it); domain adapters already
   resolve DOIs, so the JATS resolution reuses them.
3. **Contract**: markdown with fused `$…$`/`$$… \tag{N}$$` spans flows to
   the LLM consumer — token budgets shrink (no `?`-glyph noise) and
   downstream embedding quality rises measurably (that is the corpus
   result, not a hope).
4. **Docs**: QUICKREF document section + skills note; `store=True,
   include_images=True` behaviour unchanged.

## 6. Continuation: okfgraph (after gossamer route lands)

1. **`kind='jats'`** in `okfgraph/components/ingest.py`: accept a JATS
   file, delegate to the same `DocumentConverter` protocol (extend
   `converters.py` with `convert_jats(jats_path, output_dir)`,
   bobine-backed), then `_import_work_dir` as `kind='pdf'` does. The
   `honours`/ignore-params table (`_INGEST_HONOURS`) gains the `jats` row.
2. **Math-aware lint**: `_lint_converted_md` gains structural math checks —
   `$` balance after fusion, `\left`/`\right` balance, no `x^a^b`, display
   fences whole-paragraph. Auto-fix the fence/balance classes (they mirror
   P-C), never rewrite content.
3. **Math-aware chunking**: `_import_work_dir`/chunker must never split a
   `$$…$$` display across chunks; the `\tag{N}` and the enclosing section
   heading travel with the chunk that holds the formula (search context).
4. **Formula concepts (optional flag)**: one concept per display equation
   (fields: `latex`, `tag`, `section`, `paper`; edges paper→formula,
   section→formula). This makes "which paper had this inequality" a graph
   query; keep it opt-in per ingest call.
5. **Zero-code interim**: ingest `ikat/examples/auto-proof/full/*.md`
   (clean, verified math) via `kind='md'` today — useful immediately, and
   the reference point for judging bobine/`jats` output quality.

---

## 7. Non-goals

* **arXiv LaTeX as an input format** — a LaTeX parser is a project of its
  own; defer. (Callers can already pass arXiv `.tex` through the text path;
  formulas arrive intact because the source *is* LaTeX.)
* **Content MathML** (`<apply/>` trees) — publishers in scope ship
  presentation MathML; content-MathML support would be a new dispatch
  family in `mml.rs`, deferred.
* **Rendering-side OCR improvements** — P-D measures TexTeller; improving
  the model itself is out of scope.
* **The corpus glyph hacks** (`mml.py` `SPECIAL` map) — they live in the
  ikat exporter only (§2.A.2).

## 8. Ordering

P-C (sanitizer, `0.6.1`) can land immediately and independently. P-A is
the long pole (port + 511-fixture differential). P-B is small once P-A
exists. P-D reuses everything. gossamer (§5) and okfgraph (§6) start only
after `0.7.0` is released; the zero-code okfgraph interim (§6.5) can happen
any time.
