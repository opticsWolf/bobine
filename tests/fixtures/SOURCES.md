# Test-PDF corpus — provenance & attribution

The files in `tests/fixtures/pdf/` are **trimmed page ranges** extracted from
peer-reviewed preprints hosted on [arXiv.org](https://arxiv.org/). All three
source papers are licensed under **Creative Commons Attribution 4.0
International (CC BY 4.0)**, which permits redistribution, adaptation and
commercial use **provided attribution is given**. This file provides that
attribution and records the modifications we made (page trimming), as
required by CC BY 4.0 §3(a).

The full, untrimmed PDFs are **not** tracked in git — they live in
`tests/fixtures/full_pdfs/` (git-ignored) for local testing only.

---

## solitons.pdf

- **Title:** Vector Edge Solitons and Domain Walls in a Nonlinear Mechanical
  Topological Insulator
- **Authors:** David D. J. M. Snee, Yi-Ping Ma
- **arXiv:** https://arxiv.org/abs/2608.06342 (v1)
- **Source PDF:** https://arxiv.org/pdf/2608.06342
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  (verified on the abstract page)
- **Modification:** pages 1–4 only (of 18); no content altered.

## splitting_methods.pdf

- **Title:** Splitting methods for Intermediate Long Wave and perturbed
  Benjamin–Ono models
- **Authors:** Yvonne Alama Bronsard, Clémentine Courtès, Benjamin Melinand
- **arXiv:** https://arxiv.org/abs/2608.05540 (v1)
- **Source PDF:** https://arxiv.org/pdf/2608.05540
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  (verified on the abstract page)
- **Modification:** pages 1–5 only (of 28); no content altered.

## trust_ml.pdf

- **Title:** Learning When to Trust via Selective Context Preference
  Optimization
- **Authors:** Xian Sun, Wei Chow, Yingshuo Wang, Junhao Liu, Wei Gao,
  Qing Wu, Lingdong Kong
- **arXiv:** https://arxiv.org/abs/2608.06377 (v1)
- **Source PDF:** https://arxiv.org/pdf/2608.06377
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  (verified on the abstract page)
- **Modification:** pages 6–7 only (of 30); no content altered.

---

## scanned_page.pdf

Generated in-repo (MIT/Apache-2.0, bobine's own license) by
`tests/fixtures/generate_corpus.py` — a deterministic raster page with **no
text layer**, designed to exercise the scanned-document ONNX layout+OCR path.
Not from arXiv.

**Never-mode note (pdf_oxide 0.3.78):** pdf_oxide removed its
`[OCR REQUIRED]` skipped-page annotation, so the Never fast path with images
off yields empty markdown for this fixture (previously the explicit placeholder).
With images on, the page render embeds as a figure. AUTO/SURGICAL routing is
unaffected — scanned detection still sends the page through ONNX OCR.

## Regeneration

`uv run --with reportlab python tests/fixtures/generate_corpus.py`
recreates `scanned_page.pdf` deterministically. The arXiv trims can be
re-created from the source PDFs above with pdf_oxide's
`extract_page_ranges_to_bytes` (see the modification notes for ranges).

---

## ruled_table.pdf

- **Origin:** Synthetic — generated in-repo by
  `cargo run --release --example make_table_fixture` (pdf_oxide DocumentBuilder).
- **Purpose:** Regression fixture for the structured-table cascade
  (`ConverterConfig::structured_tables`). Contains a fully ruled 5×4 data grid
  with a header row, surrounded by prose paragraphs.
- **Known caveat (pdf_oxide ≤0.3.77):** pdf_oxide's spatial grid detector returned a
  degenerate grid for this page (prose sliced into columns, body rows
  concatenated), so bobine's `is_plausible_table` gate rejected it and the text
  dump path served the region.
- **Update (pdf_oxide 0.3.78):** the detector now returns a real grid — the
  header row plus the first data row (`A-101`) extract cleanly (see
  `tests/golden/ruled_table.golden.md`). Trailing rows still fragment
  (`A-102`, split `B-201` cells), so the page remains a partial-extraction
  fixture; revisit when the detector handles multi-row ruled grids.
- **Platform divergence (0.3.78, deterministic per OS — 30/30 identical runs
  on Windows):** row-banding splits the 5-row grid differently per OS.
  Ground truth is header + `A-101` (12.4/3.51/88.2) + `A-102`
  (13.0/3.77/91.5) + `B-201` (9.8/2.44/76.0) + `B-202` (10.2/2.60/79.8)
  (see `examples/make_table_fixture.rs`). Windows structures header+`A-101`
  but **drops 5 values** (`76.0`, `B-202`, `10.2`, `2.60`, `79.8` — silent
  data loss); Linux structures `B-201`+`B-202` perfectly and preserves all
  20 values (fragments the top rows instead). Both variants are pinned
  (`ruled_table.golden.md`, `ruled_table.linux.golden.md`, selected by
  `target_os` in `test_golden.rs`); any third output trips the wire.
  Upstream issue to file: same input bytes → different spans recovered per
  OS (not HashMap ordering — deterministic per process on each side).

## office/ — generated OOXML fixtures (no attribution needed)

`basic.docx`, `types.xlsx`, `deck.pptx` are **generated**, not redistributed:
`cargo run --example gen_office_fixtures` builds them from office_oxide's own
`create` API (IR → OOXML). Contents: headings, styled runs, hyperlink, bullet
+ numbered lists, 3×3 table with merged cell, footnote, embedded PNG
(checkerboard, alt text); two-sheet workbook with typed cells (unicode,
formula, date, percent, merged header, empty row) + anchored picture;
three-slide deck with bullets, table, picture. Legacy doc/xls/ppt have no
creation API — no fixtures yet.
