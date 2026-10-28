# Handoff: JATS ingest + math-aware pipeline for okfgraph (bobine proposal §6)

**From:** bobine 0.7.1 (`dev` branch; JATS adapter P-B shipped in 0.7.0,
harness + findings in 0.7.1)
**To:** okfgraph implementer (repo at `D:/User/Documents/Python/OKFgraph`,
current version 0.10.2; intended okfgraph-side copy:
`planning/handoff_bobine_jats.md` alongside the gossamer handoff)
**Status:** bobine side is done and dependency-ready. §6.5 (zero-code
interim) is **already executed** (see §7). Nothing here requires bobine
changes unless noted as an open question.
**Date:** 2026-10-28.

Related: `docs/handoff_gossamer_jats.md` (same folder, bobine repo) covers
proposal §5 — the gossamer JATS-first route that produces the XML files
this handoff consumes. The two are independently implementable: §6.1 works
on hand-supplied `.xml`/`.jats` files with no gossamer involvement.

---

## 1. Purpose and scope

PDF ingest (`kind='pdf'` → `BobineConverter` → pdf_oxide/OCR markdown)
inherits every PDF pathology, worst of all in formulas: `?` glyphs,
shredded commands, unbalanced `$`. Publisher JATS XML converts to clean,
TeX-verified markdown instead (bobine `convert_jats`, no models, no ONNX,
0.11 s for a 511-formula paper). This handoff wires that path into the
graph and then hardens the pipeline stages math actually flows through:

1. **§6.1 `kind='jats'`** — accept a JATS file, convert via a new
   `JatsConverter` (same `DocumentConverter` protocol), then run the
   standard `_import_work_dir` path exactly as `kind='pdf'` does.
2. **§6.2 Math-aware lint** — structural math checks in
   `_lint_converted_md` (and the `_str` twin); auto-fix only the
   fence/balance classes that mirror bobine's P-C sanitizer, never
   rewrite formula content.
3. **§6.3 Math-aware chunking** — never split a `$$…$$` display across
   chunks; the `\tag{N}` and the enclosing section heading travel with
   the formula chunk.
4. **§6.4 Formula concepts (opt-in flag)** — one concept per display
   equation (`latex`, `tag`, `section`, `paper`; paper→formula edges),
   making "which paper had this inequality" a graph query.
5. **§6.5 Zero-code interim** — ingest the ikat ground-truth markdown
   via `kind='md'` as the standing clean-math reference. **Done.**

**Non-goals:** touching the PDF path, the `structured` payload, the
embedding model/space, or mordant's Rust core (all §6.2–6.3 work is
Python-side); arXiv-LaTeX input; Content MathML.

---

## 2. bobine API contract for okfgraph's needs

Floor version: **`bobine>=0.7.0`** (okfgraph `pyproject.toml` currently
says `bobine>=0.6` — bump it; `convert_jats` + `JatsDocument` do not
exist in 0.6.x). Reference: bobine `python/bobine/__init__.pyi`,
`src/py_bindings.rs`, `src/jats.rs`.

```python
import bobine
doc = bobine.convert_jats(jats_path: str, work_dir: str,
                          config: Optional[ConverterConfig] = None)
doc.md_path     # str  <work_dir>/<stem>.md  (written by the call)
doc.title       # str  first '#…' line, marker stripped
doc.n_formulas  # int  MathML-backed formula elements (see §2 semantics)
doc.n_figures   # int  <fig> floats encountered
doc.n_tables    # int  <table-wrap> grids encountered
```

- **No models, no ORT, no network.** Verified: `convert_jats` runs with
  `ORT_DYLIB_PATH` unset. Conversion is pure string/XML work.
- **Fast.** 0.11 s for math-hilfer-2024 (511 formulas). No budget impact.
- **Output markdown shapes** (frozen golden:
  bobine `tests/golden/jats-chem-olive-2024.md`): `#` title, `##`/`###`
  sections, fused `$…$` inline, `$$… \tag{N}$$` displays (JATS `<label>`
  verbatim), GFM tables, `![alt](remote-href)` figures with `xlink:href`
  preserved, `[ [`ref01`]]` citation xrefs (stable, greppable — do not
  post-process).
- **Guarantees / errors** (`RuntimeError`, stable prefixes — map, don't
  string-match beyond the prefix):
  - `JATS error: XML parse: …` → malformed XML (JSON error, as today).
  - `JATS error: not a JATS article` → root isn't `<article>` (an RSS
    feed or CrossRef record fed by mistake). For `kind='jats'` this is a
    hard error (the caller named the kind; §3.5 — no sniffing fallback).
  - Prose segments are dollar-balance-checked (`check_dollars`) — output
    is balanced or the call fails. There is no silent-unbalanced case.
  - DTDs (`<!DOCTYPE article PUBLIC …>`) are stripped; JATS 1.1/1.2/1.3
    all parse (local-name matching, namespaces ignored).
- **Counts semantics for provenance (observed):** `n_formulas`
  increments for MathML-backed `<inline-formula>` / `<disp-formula>`
  elements with non-empty TeX — *not* for inline `$…$` spans
  synthesized from non-MathML markup (sub/sup runs), and *not* a
  display-only count either (chem-olive: 2 displays → `n_formulas=2`;
  math-hilfer: 511 displays → 511). Record the raw counts, don't
  reinterpret them; re-verify per bobine version.

---

## 3. §6.1 `kind='jats'` — implementation

### 3.1 `JatsConverter` (in `okfgraph/components/converters.py`)

New class beside `BobineConverter`, satisfying `DocumentConverter`
**structurally** (Python `Protocol` — no inheritance needed):

```python
class JatsConverter:
    """JATS-XML implementation of DocumentConverter (bobine>=0.7.0)."""

    def convert(self, pdf_path, output_dir, *, on_page=None):
        import bobine  # lazy, same pattern as BobineConverter
        doc = bobine.convert_jats(str(pdf_path), str(output_dir))
        image_dir = Path(output_dir) / "assets"
        image_dir.mkdir(parents=True, exist_ok=True)
        if on_page is not None:
            on_page(0, 1)
        return ConvertedDocument(
            md_path=Path(doc.md_path),
            image_dir=image_dir,   # always empty: JATS figures are remote hrefs
            page_count=0,          # JATS has sections, not pages
        )
```

Three deliberate decisions, all spec — implement as stated:

1. **Separate class, not a `BobineConverter` mode.** Routing/extraction
   options are meaningless for XML; overloading them invites
   `routing_mode="always"` on a `.xml` file doing nothing silently.
2. **The parameter keeps the protocol name `pdf_path`** even though it
   takes a JATS path. Renaming it breaks structural conformance for
   positional callers (`converter.convert(path, work_dir)` in
   `ingest.py`). Document the wrinkle in the docstring; do not "fix" it.
3. **`page_count=0`, empty `image_dir`.** Downstream (`_import_work_dir`,
   chunking, `ConvertedDocument` consumers) must treat 0 as "unpaged",
   not as failure. Grep for `page_count == 0` assumptions before
   implementing — if any consumer divides by it or refuses 0, that is a
   pre-existing bug this handoff exposes; fix the consumer, not the
   converter. Same for "empty image_dir means extraction failed"
   heuristics: JATS figures are remote `href`s by design (nothing to
   stage, nothing to embed — see §3.5).

Also update `docs/converters.md` (the seam doc quotes the protocol):
add the JATS row — same contract, `page_count=0`/empty-images semantics.

### 3.2 `ingest(kind='jats', …)` (in `okfgraph/components/ingest.py`)

- `_INGEST_HONOURS["jats"] = frozenset({"jats_path", "output_dir"})`.
  New sibling param `jats_path` on `ingest()` (do **not** reuse
  `pdf_path` — the honours-table error messages name the ignored param,
  and `kind='jats' ignores: pdf_path` is exactly the guidance a caller
  needs). The existing `ignored`-detection then automatically rejects
  `routing_mode`/`extract_images`/`mode` for jats with `BAD_VALUE` —
  which is correct: image-ingestion `mode` is inapplicable (remote
  figures, §3.5) and routing is meaningless.
- New `if kind == "jats":` branch mirroring the `pdf` branch step for
  step: `_require_file(jats_path)` → `_resolve_converter(...)`
  (extended with kind-awareness — its current signature takes
  routing/extract flags that are meaningless for JATS, so either add a
  kind parameter or a small `_resolve_jats_converter` sibling; an
  explicit `converter=` instance is still honoured verbatim) → convert
  into temp dir (auto import) or `output_dir` → **`_lint_converted_md(md_path,
  auto_fix=True)`** (math-aware after §6.2) → `_import_work_dir(…)`.
- Namespace: extend `_ingest_namespace` with a JATS twin keyed
  `jats-<hash12>` of the source XML bytes (same stability/idempotence
  rationale as `pdf-<hash12>`; a PDF and its JATS twin must never share
  a namespace — hashes differ anyway, but the prefix makes debugging
  possible).
- Return dict: same keys as pdf (`md_path`, `concept_ids`,
  `image_dir`, `page_count`) **plus** `jats_title`, `n_formulas`,
  `n_figures`, `n_tables` straight from `JatsDocument`. These numbers
  are the cheapest health metric the caller gets (`n_formulas == 0` on
  a math-heavy paper ⇒ suspect input, not a clean paper).
- Final `kind must be …` error message gains `'jats'`.
- `auto_import=False` writes `output_dir` only, mirroring pdf.

### 3.3 Surfaces: CLI + MCP + skills

- CLI (`okfgraph/cli.py`, `ingest` subcommand): `choices=["md", "pdf",
  "thoughts", "bib"]` → add `"jats"`; add `--jats-path` ("JATS XML file
  (--kind jats)"); the existing `--output-dir` / `--no-auto-import` /
  `--mode`-family wiring applies unchanged. `--routing-mode` and
  `--no-extract-images` with `kind=jats` must produce the honours-table
  `BAD_VALUE`: the honours table only sees the `passed` dict
  (`md_path … bib_path`) — `routing_mode`/`extract_images`/`mode` are
  separate params, so the jats branch must reject them explicitly
  (`routing_mode != "auto"`, `extract_images is False`, or
  `mode != "text"` → `OKFError(BAD_VALUE)` naming the param).
  Mirror the pdf auto-import rule: with `auto_import=True`, `output_dir`
  is excluded from the jats honours set (explicit work-dir export only).
- MCP `ingest` tool (`mcp_server.py:302`, `kind: Literal["md",
  "pdf", "thoughts"]` — note it already lags the Python API, which
  also supports `"bib"`; add `"jats"` here, and consider closing
  the `"bib"` gap in the same change rather than repeating the
  pattern): extend the Literal, add `jats_path` + `formula_concepts`
  params with per-kind descriptions in the existing style; `data` per
  kind gains jats → `{concept_ids, jats_title, n_formulas, n_figures,
  n_tables, chunk_count, …}`. `kind="md"`/`"pdf"` shapes
  byte-identical.
- `skills/okfgraph-ingest/SKILL.md` kind table: add the JATS row
  (`kind='jats'` | JATS XML file → converted + imported; counts
  returned; remote figures not staged). One row, same style as the
  `pdf` row. `skills/okfgraph-cli/SKILL.md` only if it enumerates kinds
  (check; otherwise leave it).

### 3.4 `converter_status` / doctor

`converters.converter_status()` reports ONNX-model readiness for the PDF
path. JATS needs nothing — extend the note payload, not the logic: when
`bobine>=0.7.0` is present, `note` should mention JATS availability
(e.g. append `"; JATS conversion available"`), so `doctor` answers "can
I ingest this XML" without a model download. Version gate: `bobine`
present but `<0.7.0` → `available` stays False with the existing
upgrade note (no `convert_jats` exists there).

### 3.5 Edge cases (decided)

- **Non-JATS XML** (`not a JATS article`): hard `OKFError` for
  `kind='jats'`. The one exception: `kind='md'`-style auto-detection is
  out of scope — no sniffing dispatcher in this handoff.
- **Figures**: remote `href`s (`info:doi/…`, publisher CDN URLs) are
  preserved in the markdown and surface via `extract_links` as
  follow-up candidates for free. Never download them during ingest
  (license + size + SSRF surface). The jats return dict carries no
  image manifest (neither does pdf's) — `image_dir` is an empty staging
  dir, `page_count` is 0; document "figures are remote hrefs" in the
  skill row and CLI help instead of inventing a manifest field.
- **Re-imports**: namespace hash makes them idempotent upserts, same as
  pdf. `force` semantics unchanged.
- **Huge single-file displays**: handled at chunk time (§6.3), never at
  convert time.

---

## 4. §6.2 Math-aware lint

### 4.1 Where

`_lint_converted_md` and `_lint_converted_md_str`
(`okfgraph/components/ingest.py:292/369`) keep their mordant pass
(`MD009/MD012/MD047` fixable, `MD001/MD031/MD033` errors) **untouched**.
Add a Python `_lint_math(content) -> (fixed_content, diagnostics)` pass
after mordant, merging into the same return-dict shape with rule codes
in a new `MATH…` namespace (no collision with mordant codes, no change
to existing assertions in `tests/test_lint.py`).

Rationale for Python-side, not mordant-side: these are regex/brace
scans, not block-structure rules; no Rust changes, no wheel rebuild,
shippable in one PR. Reference implementations to mirror (bobine repo):
`src/sanitize.rs` (`split_cmds`, `fix_unclosed_fences`,
`separate_consecutive_scripts`), `src/jats.rs` `check_dollars` /
`prose_guard`.

### 4.2 The four checks (exact)

| Code | Check | Severity | Auto-fix |
|---|---|---|---|
| `MATH001` | Unbalanced unescaped `$` in the file (even count required; `\$` excluded; `$$` pairs count twice) | error, report-only | **none** — intent unknowable; fixing masks corruption |
| `MATH002` | `\left` without a matching `\right` inside one `$…$`/`$$…$$` span | error | append `\right.` before the span's closing fence — verbatim the P-C rule, mechanical and TeX-safe |
| `MATH003` | Consecutive same-level scripts `x^a^b` / `x_a_b` (regex on the span interior, braces/comments excluded) | error | `x^a{}^b` — verbatim the P-C `separate_consecutive_scripts` rule |
| `MATH004` | A `$$` display fence whose opener and closer land in different mordant blocks (pre-chunk scan: fence line-numbers vs `MarkdownChunker` block spans) | error, report-only | **none** — indicates a broken fence the converter should never emit; surfaces converter bugs as data |

Scope discipline (from the proposal, enforce in review): fixes touch
only delimiters/empty groups (`\right.`, `{}`); **formula interiors are
never rewritten**. A lint that "repairs" `\frac` argument order is a
content edit and is out of scope.

### 4.3 Acceptance

- `tests/test_lint.py`-style new cases: each check has a minimal
  failing fixture + (where applicable) the exact fixed string.
- Regression anchor: re-import the §6.5 GT concepts' markdown through
  `_lint_converted_md_str` — the TeX-verified corpus must yield **zero**
  `MATH…` diagnostics. Any hit is a lint false positive, not a corpus
  error; fix the check, not the fixture.
- bobine-side cross-check: `convert_jats` output on all 10 corpus XMLs
  must yield zero `MATH001`/`MATH004` (bobine guarantees balance); run
  once, record in the PR.

---

## 5. §6.3 Math-aware chunking

### 5.1 Measured baseline (do not re-derive — build on it)

Probed against the installed mordant (`MarkdownChunker.get_all_chunks`):

- A standalone `$$…$$` display is already its **own `Paragraph` block**
  — mordant does not merge it with surrounding prose. The common case
  is therefore already intact.
- `_split_into_chunks` (`embedding.py:1235`) already prepends the
  current section heading as ephemeral `heading_context` in the
  *embedding payload* (not stored). The proposal's "section heading
  travels with the chunk" is thus **half-done** — keep it, extend it.
- The actual split risk is `_cap_chunk_budget` (`embedding.py:33`):
  an oversized block is word-tiled with zero regard for fences, so a
  long multi-line display (routine in papers like math-hilfer-2024)
  can be cut mid-formula into `Type+` continuations.

### 5.2 Spec (minimal, in `_split_into_chunks` / `_cap_chunk_budget`)

1. **Fence-aware cuts.** Before tiling an oversized block, compute the
   `$$…$$` spans inside it (same unescaped-`$` scan as MATH001). Cut
   points may only fall *between* spans (or outside all spans); a span
   is atomic. A single span over budget is emitted whole — the
   context-window warning remains the backstop, exactly as for
   unsplittable single words today.
2. **Tag travels automatically** — `\tag{N}` sits inside the display
   block, so atomicity is sufficient. No extra code; assert it in tests
   (tag present in the same `chunk_text` as its formula).
3. **Heading: keep ephemeral (decided).** `heading_context` is
   payload-ephemeral today; baking it into stored `chunk_text` was
   considered and rejected: reconstruction (`reconstruct_document`,
   ~98% fidelity) rejoins stored text, and baked-in headings would
   double them; retrieval already embeds the heading via the payload.
   Revisit only with a measured retrieval regression on formula queries.
4. **Inline `$…$` spans** inside prose blocks: atomicity at the *span*
   level would need the same span-aware cuts (an inline span can be
   split when its *enclosing* block exceeds budget, even though the span
   itself is small). Explicitly deferred: wait for a failing test before
   paying for it.

### 5.3 Acceptance

- New tests beside `tests/test_chunk_cap.py` / `test_chunking.py`:
  a 3 k-token single display stays one chunk; a prose+display+prose
  oversized block cuts only between spans; `\tag{N}` and formula share
  `chunk_index`; heading present in the embedding payload.
- Reconstruction fidelity on a formula-heavy GT file stays ≥ the
  current ~98% (run `reconstruct_document` before/after on one §6.5
  concept and diff).

---

## 6. §6.4 Formula concepts (opt-in)

### 6.1 Shape

New ingest param `formula_concepts: bool = False` (CLI
`--formula-concepts`, MCP `ingest` field), honoured by kinds `jats`
(and later `pdf`; explicitly not `md`/`bib`/`thoughts` — free-form
markdown has no section/equation contract worth mining).

Per display span in the *post-lint* markdown (`$$…$$` regex over the
imported content — the same scan as MATH001, reused not reinvented):

- **ID:** `{paper-concept-id}/formula/{slug}` where slug is the
  `\tag{…}` contents slugified, else the 1-based display ordinal
  (`eq-1`, `eq-2`, …). Paper concept id is the namespaced id the work
  dir import mints (e.g. `@jats-<hash12>/…` stem). Deterministic ⇒
  re-imports upsert.
- **Type/frontmatter** (`ConceptModel` allows arbitrary keys —
  no schema change):
  `type: "formula"`, `title` = tag or first 60 chars of LaTeX,
  `body` = the display block verbatim + one line `Section: …` context,
  frontmatter `latex` (full span, fences included), `tag`, `section`
  (nearest preceding `##+` heading), `paper` (paper concept id).
- **Edges:** wikilinks are the edge mechanism (`extract_wikilinks`) —
  the formula body links `[[paper-id]]` (formula→paper) and, where the
  section heading maps to an existing section concept, `[[section-id]]`.
  Paper→formula discovery runs through backlink search over those links
  (verify the backlink query path exists during implementation; if the
  graph needs explicit reverse edges, file it as a follow-up — do not
  mutate paper bodies, which would perturb hashes/deltas).
- **Counts:** return `formula_concept_ids` alongside `concept_ids`;
  `chunk_count` covers them like any other concept.

### 6.2 Costs and guardrails (state in docs)

- One concept + embedding per display: a 511-display paper mints 511
  concepts. Default **off**; the flag help text says so.
- Tag-less displays get ordinal slugs — stable within a source version,
  not across reconversions that reorder equations. Document that;
  publisher `<label>`s (the common case in JATS) are stable.
- `prune_missing`/`force` semantics inherit from the normal import;
  stale formula concepts from a removed equation prune like any other
  concept.

### 6.3 Acceptance

- chem-olive import with the flag: `len(formula_concept_ids)` equals
  the number of `$$…$$` display spans in the post-lint markdown
  (2 for chem-olive), each concept's `latex` fence-balanced (run the
  §6.2 scan over them), each links its paper, search for a distinctive
  symbol (e.g. `CdCl`) surfaces the formula concept.
- Flag off (default): byte-identical import to today (no new concepts,
  no new frontmatter keys on paper concepts).

---

## 7. §6.5 Zero-code interim — DONE (reference, not work)

Executed 2026-10-28 against the local graph (`kb.db`): all 10 ikat
ground-truth markdowns ingested as `ikat-ground-truth/<paper>`
(~1,425 chunks), search-verified (`PRODY` resolves to the econ-tax GT
concept), logged under `thoughts/bobine-analysis`. Standing uses:

- The §6.2 regression anchor (GT markdown must yield zero `MATH…`).
- The §6.3 reconstruction baseline (formula-heavy GT file).
- The human reference for "is the JATS route better than the PDF
  route" on any paper in the set.

Do not re-ingest; do not move them. If the GT files change upstream,
re-import and note the new content hash in the PR.

---

## 8. Testing matrix

| Area | Location | Must cover |
|---|---|---|
| JATS converter | `tests/test_converter.py` style new file | `JatsConverter.convert` returns protocol shape; `page_count==0`; empty `image_dir` created; missing file → `FILE_NOT_FOUND` *before* bobine runs; `bobine<0.7.0`/absent → actionable error (mirror `converter_status` tests) |
| `kind='jats'` dispatch | `tests/test_ingest.py` style | honours-table rejects `routing_mode`/`extract_images`/`mode` with `BAD_VALUE`; `auto_import=False` writes `.xml`+`.md` only; return dict carries the four JATS counts; idempotent re-import (same ids) |
| Math lint | `tests/test_lint.py` style | four checks, exact fix strings for MATH002/MATH003, GT-zero-findings anchor |
| Chunking | `test_chunk_cap.py`/`test_chunking.py` style | atomic displays, tag co-location, payload heading, reconstruction fidelity |
| Formula concepts | new file | counts, links, search hit, flag-off byte-identity |
| CLI/MCP | `test_ingest_tool.py`, CLI tests | `--kind jats --jats-path`, `--formula-concepts`, MCP param shapes |

Corpus fixtures (bobine repo, vendor don't duplicate): smallest JATS
`ikat-corpus/jats/chem-olive-2024.xml` (62 KB) for unit tests;
`med-ioannidis-2005.xml` (80 KB) as the second shape (0 formulas,
2 figures per the census — exercises the near-empty math path). Expected chem-olive counts:
`n_formulas=2, n_figures=7, n_tables=2` (verified against bobine 0.7.1;
re-verify on bump — inline spans from non-MathML markup are not counted,
§2).

---

## 9. Docs to update (okfgraph side)

1. `docs/converters.md` — JATS row in the seam contract
   (`page_count=0`, empty-images semantics, `JatsConverter`).
2. `skills/okfgraph-ingest/SKILL.md` — kind-table JATS row (+ the
   `--formula-concepts` flag where flags are listed); the MCP `ingest`
   section's per-kind `data` shapes.
3. CLI `--help` strings (they *are* user docs — write the
   remote-figures note into `--jats-path`/store help).
4. `CHANGELOG.md` — one entry per sub-section shipped (§6.1 … §6.4
   land independently; §6.2 and §6.3 pair well in one release since
   they share the fence-scan).
5. This handoff's gossamer twin stays in the bobine repo; link it from
   the okfgraph PR description, not by copy.

---

## 10. Ordering, dependencies, open questions

**Order:** §6.1 → §6.2 → §6.3 → §6.4. Each lands and releases
independently; §6.2's GT anchor needs only the §6.5 concepts (present).
§6.4 needs §6.3's atomicity (formula regexes assume intact displays).

**Dependencies:** `bobine>=0.7.0` (pyproject floor bump). No mordant
changes, no Rust changes, no gossamer changes. The gossamer §5 route
is a *producer* of JATS files, not a prerequisite — hand-supplied XML
unblocks everything here.

**Open questions for the implementer (decide, don't defer silently):**

1. Backlink query path for paper→formula (§6.1): verify it exists;
   else file the reverse-edge follow-up.
2. `ConvertedDocument.page_count=0` consumers: grep first; fix any
   `==0`-as-failure assumption at the consumer.
3. Stored-vs-ephemeral heading for formula chunks (§5.3 took ephemeral
   — revisit only on measured retrieval loss).
4. Ordinal slugs for untagged displays: acceptable instability, or
   content-hash slugs? (Hash slugs are stable but unreadable; pick one
   and document it.)
5. `page_count` in the jats return dict: 0 (honest) vs section count
   (useful)? Recommend 0 — sections are not pages, and chunk counts
   already measure size.
6. Should `kind='pdf'` later gain the same `formula_concepts` flag?
   Yes, intended — but only after PDF math quality is trustworthy
   (bobine's P-D renderer blocker is still open upstream; mining
   formulas from OCR-mojibake would enshrine garbage as concepts).
