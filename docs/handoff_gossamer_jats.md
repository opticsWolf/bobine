# Handoff: JATS-first document route for gossamer (bobine proposal §5)

**From:** bobine 0.7.1 (`dev` branch; JATS adapter P-B shipped in 0.7.0)
**To:** gossamer implementer (repo at `D:/User/Documents/Python/gossamer`,
target branch `dev_rust`; intended gossamer-side copy:
`planning/handoff_bobine_jats.md`)
**Status:** bobine side is done, tested, and dependency-ready. Nothing in
this handoff requires bobine changes unless noted as an open question.
**Date:** 2026-10-28. Endpoint liveness re-verified this date (see §8).

---

## 1. Purpose and scope

When gossamer extracts a scholarly paper today, it downloads the PDF and
parses the text layer / OCR. That path inherits every PDF pathology:
ligature mojibake, hyphenation fragments, and — decisive for math — formulas
arrive as glyph soup (`?` placeholders, unbalanced `$`). The ikat corpus
proves the alternative: for the same 10 papers, publisher JATS XML converts
to clean markdown with fused `$…$` inline spans and
`$$tex \tag{N}$$` displays, all TeX-verified against publisher MathML.

**Scope of this handoff (proposal §5 only):**

1. `DocumentExtractor.extract_document` (+ MCP/CLI surface): add a
   structured-source step *before* the pdf_oxide path — paper URL/DOI →
   `OpenAccessLocator` → JATS XML (EuropePMC / PMC / PLOS) → fetch →
   delegate to `bobine.convert_jats` → return gossamer `ExtractionResult`.
   Fall back to the current PDF path when no structured source resolves.
2. Config: `prefer_jats: bool = True` (default **on**; §3.4 explains why
   not `prefer_structured`).
3. Downstream contract: markdown with fused math spans flows to the LLM
   consumer (smaller, cleaner, embedding-friendlier).
4. Docs: QUICKREF document row + scholarly-pipeline line + skills note;
   `store` / `include_images` behaviour unchanged in spirit (details §3.6).

**Non-goals:** okfgraph `kind='jats'` + math lint/chunking/formula concepts
(proposal §6 — separate handoff, okfgraph side); arXiv-LaTeX input;
Content MathML; touching `extract_document_structured` or the PDF path.

---

## 2. bobine API contract (exact, verified against 0.7.1)

### 2.1 Dependency

- PyPI name: **`bobine`**. Floor version: **`bobine>=0.7.0`**
  (`convert_jats` + `JatsDocument` shipped in 0.7.0; 0.7.1 only adds
  unrelated media options).
- Import cost is negligible and **the JATS path needs no models and no
  ONNX Runtime**: verified `bobine.convert_jats` on a corpus XML with
  `ORT_DYLIB_PATH` unset (no dylib, no downloads, no GPU/CPU EP).
  Recommend a lazy `require_bobine()` mirroring `require_pdf_oxide()`,
  with an actionable error naming the `bobine>=0.7.0` extra — but unlike
  `pdf-oxide`/`office-oxide`, this can live in core deps, not an extra,
  because it never needs native model weights. (The Rust extension wheel
  itself is still maturin-built; standard pip install.)
- Reference: `D:/User/Documents/Rust/bobine/python/bobine/__init__.pyi`
  (stubs), `src/py_bindings.rs` (`convert_jats`, `PyJatsDocument`),
  `src/jats.rs` (converter).

### 2.2 Functions

```python
# One-shot (recommended for gossamer): no converter object needed.
doc: JatsDocument = bobine.convert_jats(path: str, work_dir: str,
                                        config: Optional[ConverterConfig] = None)

# Method form (same semantics; needs a cache dir for the constructor even
# though the JATS path never touches models — pass the gossamer cache dir).
conv = bobine.HybridConverter(bobine.ConverterConfig(), cache_dir)
doc: JatsDocument = conv.convert_jats(path, work_dir)

# The generic dispatcher also routes by extension (verified src/converter.rs:982):
md: str = conv.convert(path_with_xml_or_jats_suffix, work_dir)
```

`JatsDocument` properties: `md_path: str` (`<work_dir>/<stem>.md`,
written by the call), `title: str` (first `#` line, marker stripped),
`n_formulas: int` (inline + display), `n_figures: int` (`<fig>` floats),
`n_tables: int` (`<table-wrap>` grids).

Errors surface as `RuntimeError` with stable prefixes — map them to
gossamer's JSON error schema, do not string-match beyond the prefix:

| bobine message | meaning | gossamer mapping |
|---|---|---|
| `I/O error: … (os error 2)` | missing file | same as today's missing-local-file error |
| `JATS error: XML parse: …` | malformed XML | `Document extraction failed: …` (verbatim) |
| `JATS error: not a JATS article` | root element isn't `<article>` (e.g. RSS/CrossRef/TEI fed by mistake) | fall back to the PDF path, **not** a hard error (see §3.5) |

### 2.3 Output markdown contract (what the LLM consumer gets)

Crib from the frozen golden `tests/golden/jats-chem-olive-2024.md`
(bobine repo) — pin gossamer tests to the same shapes:

- `# {title}` then `##` sections / `###` subsections from JATS `<sec>`.
- Inline math fused: `…sulfonamide group (SO$_{2}$NH) and…`
  (MathML → LaTeX via `mml.rs`, then `merge_math` fuses adjacent spans).
- Display equations: `$$… \tag{Eq 1}$$` / `$$… \tag{1}$$` — the JATS
  `<label>` becomes the `\tag{}` verbatim.
- Tables: GFM pipes, caption preserved; figures:
  `![Fig 1: Schematic representation …](info:doi/10.1371/…​.g001)` —
  **remote `xlink:href` URLs are preserved as-is** (no download, no
  rewrite). In-text `[(**Fig** 1)]`-style xrefs stay as literal text.
- Citation xrefs render as `[ [`ref01`]]` (double-bracket + code span) —
  ugly but stable and greppable; do **not** post-process them (they anchor
  GT comparisons).
- **Guarantee:** `check_dollars` runs over every prose segment — output is
  either dollar-balanced or the conversion errors. There is no silent
  unbalanced-`$` case to defend against downstream.
- Input tolerance: `<!DOCTYPE article PUBLIC …>` DTDs are stripped
  (roxmltree rejects DTDs); JATS 1.1/1.2/1.3 all parse (matching is on
  local names, namespaces ignored, `xlink:href` found by suffix match).
- Performance: 0.11 s for the largest corpus paper (511 formulas,
  math-hilfer-2024) — pure CPU string/XML work. No budget impact.

### 2.4 Corpus evidence (cite in tests and docs)

`ikat-corpus/` in the bobine repo (fixtures, not shipped to crates.io):

- `jats/*.xml` — 10 publisher JATS files (PLOS family + PMC samples).
  Smallest: `chem-olive-2024.xml` (62 KB — **vendor this one** into
  gossamer `tests/fixtures/`), `med-ioannidis-2005.xml` (80 KB).
- `ground-truth/*.md` — TeX-verified markdown per paper (the reference
  point for "is the JATS route better than the PDF route").
- Display/inline census per paper (displays, inlines):
  bio-medsea `0/14`, chem-olive `2/7`, cs-bodvw `9/6`, earth-ph `0/4`,
  econ-tax `28/0`, eng-disk `29/4`, math-hilfer `511/4`,
  med-ioannidis `0/2`, physics-pauli `113/8`, psych-maia `0/1`.
- Differential gate: `tests/test_mml.rs` (692/692 MathML→LaTeX fixtures
  byte-identical to the Python reference) and `tests/test_jats.rs`
  (census + `figures_link_out` + `display_equations_carry_tags` +
  frozen golden). Gossamer needs only the chem-olive XML + its expected
  counts (`n_formulas=9, n_figures=7, n_tables=2`; title starts
  `Olive leaf extract-assisted green synthesis`).

---

## 3. gossamer implementation

### 3.1 Where the route lives

In `gossamer/document.py`, `DocumentExtractor.extract_document`, insert
the JATS attempt **after** URL validation/robots/rate-limit gating and
**before** `normalize_url`/cache lookup — it must flow through the same
`_tb` machinery (SSRF guard, robots, per-domain rate limit, budget,
guard via `_finish_document`). Pseudocode:

```python
def extract_document(self, source, pages=None, structured=False, *,
                     store=False, store_dir=None, include_images=False,
                     tables_as="json", prefer_jats=True):
    ...
    if prefer_jats and not structured and pages is None:
        jats_hit = self._try_jats_route(source, store=store,
                                        store_dir=store_dir)
        if jats_hit is not None:
            return jats_hit  # ExtractionResult JSON, already guarded
    ...  # existing PDF/Office path unchanged
```

`_try_jats_route(source, …) -> Optional[str]` returns the finished JSON
string on success, `None` when the source is not JATS-resolvable
(caller falls through silently). Hard errors (network failure *after* a
JATS URL resolved, `JATS error: XML parse`) return the JSON error —
same policy as the PDF path.

### 3.2 DOI acquisition (reuse, don't reinvent)

- Bare DOI / `doi:` / `doi.org` URL → existing
  `open_access.normalize_doi` (it validates shape only — keep that).
- `https://doi.org/…` or publisher landing URL → normalize to DOI first
  (strip query/fragment; `normalize_doi` already rejects those — catch
  `ValueError` → not JATS-resolvable → fall through).
- **Local PDF file** → existing `agent_tools._doi_from_pdf`
  (`citations.find_doi_in_text`, first DOI in text). Local `.xml`/`.jats`
  file → attempt `convert_jats` directly; `not a JATS article` → fall
  through to the normal path (an `.xml` feed must keep working).
- Non-DOI, non-resolvable URLs (arXiv abs/pdf, generic pages) → `None`
  immediately, zero extra requests. (arXiv Vanity/CEUR HTML is out of
  scope; the PDF path already serves arXiv.)

### 3.3 JATS resolution order (verified 2026-10-28)

Resolve DOI → JATS URL in this order; first success wins. All fetches go
through `_fetch_document_url`-equivalent plumbing (headers, timeout,
SSRF re-validation of the *resolved* URL — it is third-party data).

1. **EuropePMC `fullTextXML` (primary).** One API, all publishers,
   single file, no unpacking:
   - Search: `GET https://www.ebi.ac.uk/europepmc/webservices/rest/search`
     `?query=DOI:{doi}&format=json` → **verified HTTP 200 today**.
     Take `resultList.result[0]` with `source == "PMC"` (else `inPMC ==
     "Y"`); PMCID = `"PMC" + id` (accept an already-prefixed `id`
     defensively — **assert the exact shape in tests against the live
     response**, field names above are from the documented schema).
     No hit → `None` (fall through).
   - Fetch: `GET https://www.ebi.ac.uk/europepmc/webservices/rest/`
     `{source}/{id}/fullTextXML` → JATS XML (`application/xml`).
     Assert `Content-Type` contains `xml` and body starts with `<`
     before handing to bobine; else `None`.
   - Politeness: append `&email={operator}` from the existing OpenAlex
     email config when present (EuropePMC requests it for heavy use);
     the per-domain rate limiter already covers pacing. No key needed.
2. **PMC OA API (fallback).** Only when EuropePMC has no full text:
   `GET https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={PMCID}` →
   **verified HTTP 200 today** → parse `<record><link format="tgz"
   href="…tar.gz"/></record>` → download → unpack → take the `*.nxml`
   (exactly one per package; if several, prefer `*.nxml` whose basename
   matches the PMCID). Needs `tarfile` (stdlib) — no new dep.
   No `<link>` → `None`.
3. **PLOS `?type=manuscript` (last resort).**
   `GET https://journals.plos.org/{journal}/article/file?id={doi}&type=manuscript`
   serves the manuscript XML — but a status probe **returned HTTP 403
   today** (bot mitigation; a full GET with browser headers may still
   succeed). Attempt it, but treat any non-200 as `None`, never an error.
   The journal path segment (`plosone`, `plosbiology`, …) is not
   derivable from the DOI — get it from the `OpenAccessLocator`
   landing-page URL when it is a `journals.plos.org` host, else skip.

Do **not** add Unpaywall as a JATS source: it resolves landing pages,
not full-text XML, and costs an email-configured request for no gain
(the OA PDF candidates it yields are already consumed by `locate-pdf`).

### 3.4 Signature and naming

```python
def extract_document(self, source, pages=None, structured=False, *,
                     store=False, store_dir=None, include_images=False,
                     tables_as="json", prefer_jats=True) -> str: ...
```

- Name is deliberately **`prefer_jats`**, not the proposal's
  `prefer_structured`: `structured=True` already means "structured
  payload (metadata/tables/links)" and the two flags must compose
  without confusion. Default `True` (proposal §5.2: corpus justifies it).
- MCP `extract_document` tool + `agent_tools.extract_document_structured`
  pass-through: add optional `prefer_jats` (default True) to the tool
  schema; `extract_document_structured(source)` stays PDF-only/unchanged.
- CLI: `extract … [--prefer-jats | --no-prefer-jats]` (default on).
  No change to `--pages/--structured/--store/--include-images/--tables-as`
  spellings.

### 3.5 Flag interactions (all decided — implement as stated)

- `pages="10-20"` + JATS resolvable → **actionable error**
  (`pages=… is not supported for structured-source (JATS) extraction;
  re-run without pages or with --no-prefer-jats`). JATS has sections,
  not pages; silent page-dropping would corrupt citations.
- `structured=True` → JATS route **skipped** (structured payload stays
  PDF-backed). Rationale: separate feature, separate tests.
- `tables_as != "markdown"` → JATS route skipped unless `"markdown"`.
  JATS tables are already GFM pipes; transcoding them to JSON/CSV is a
  new feature, not this handoff. (With `"markdown"` — the common case —
  the route applies.)
- `store=True` → store **two** files: the fetched XML verbatim
  (`<stem>.xml`) + the converted markdown (`<stem>.md`); reuse
  `_store_document` with the XML bytes. `include_images=True`:
  figures are remote `href`s (nothing embedded to save) → succeed as a
  **documented no-op** and note it in the `stored.resources` manifest
  (`{"embedded": [], "note": "JATS figures are remote hrefs, not saved"}`).
  Do not download remote figures (license + size + SSRF surface).
- `JATS error: not a JATS article` on a `.xml`/`.jats` source → fall
  back to the normal path (feeds must keep working). Any other bobine
  error → JSON error, same shape as PDF-path failures.

### 3.6 Caching and provenance

- Cache key: `{url}#jats` (never collide with the PDF rendering of the
  same DOI/URL). Cache the **untruncated markdown**; truncation,
  token-count, and link extraction run on read, exactly like today.
- `ExtractionResult` provenance: `final_url` = the fetched JATS URL,
  `content_type` = response content type, `http_status` set; add extras
  (pydantic `extra` or new optional fields — your call, but keep them):
  `resolved_doi`, `jats_source` (`europepmc`|`pmc-oa`|`plos`),
  `jats_title`, `n_formulas`, `n_figures`, `n_tables`.
  These numbers are the cheapest health metric the LLM consumer gets
  (`n_formulas == 0` on a math-heavy paper ⇒ suspect rendering).
- `links=extract_links(full_content)` runs unchanged — figure `href`s and
  `info:doi/…` strings flow into follow-up candidates for free.
- `store` writes are **not** cached (existing policy); the XML bytes for
  `store` come from the fetch in hand, no refetch.

### 3.7 What stays untouched

`_download_and_extract`, `_parse_document_bytes`, page-range machinery,
the guard (`_finish_document`), robots/SSRF/rate-limit helpers, the
`documents` extra (`pdf-oxide`/`office-oxide` remain the PDF path),
`locate-pdf`, `cite`, keystore, and the Rust `_core` (no Rust changes —
bobine is a pure-Python-side dependency).

---

## 4. Downstream contract (for docs + skills note)

> When the source paper has publisher full text (most PLOS / PMC content),
> `extract` returns markdown converted from JATS XML: section headings,
> GFM tables, figures as `![alt](remote-href)`, inline math `$…$` and
> display math `$$… \tag{N}$$` with publisher equation numbers intact.
> Dollar spans are guaranteed balanced (conversion fails otherwise), so
> downstream chunking must still never split a `$$…$$` display — but it
> no longer needs to repair OCR-mojibake. Provenance fields
> (`jats_source`, `n_formulas`, …) say which route served the request;
> absence of `jats_source` means the PDF path served it.

Token-budget note (for QUICKREF): JATS markdown is typically 10–30%
shorter than the same paper's PDF extraction (no hyphenation fragments,
no `?`-glyph noise, no repeated running heads) — and the math is
*correct*, which is the actual win for embedding quality.

---

## 5. Testing (gossamer side)

- **Offline unit (no network, no models):** vendor
  `bobine-repo/ikat-corpus/jats/chem-olive-2024.xml` → 
...[truncated 3074 chars]