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

## Regeneration

`uv run --with reportlab python tests/fixtures/generate_corpus.py`
recreates `scanned_page.pdf` deterministically. The arXiv trims can be
re-created from the source PDFs above with pdf_oxide's
`extract_page_ranges_to_bytes` (see the modification notes for ranges).
