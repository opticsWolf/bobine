"""Generate the deterministic raster (scanned-like) PDF fixture.

The arXiv fixtures in ``tests/fixtures/pdf/`` are born-digital PDFs with a
text layer — they exercise the fast path, SURGICAL formula pass and layout
detection. This generator produces the one page type arXiv can't provide:
a **scanned-like page** — pure raster, no text layer — which drives the
``_is_scanned`` → ONNX layout+OCR path (routing modes SURGICAL/ALWAYS).

The output is committed as a golden fixture (``scanned_page.pdf``); this
script is kept for regeneration/audit. Requires ``reportlab`` + ``Pillow``:

    uv run --with reportlab python tests/fixtures/generate_corpus.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIXTURES = Path(__file__).parent / "pdf"
OUT = FIXTURES / "scanned_page.pdf"

# A page of plausible academic text the OCR engine can actually read.
_PARAGRAPHS = [
    "The rapid advancement of document understanding systems has",
    "enabled automated extraction of text, tables and mathematical",
    "expressions from scientific literature. In this paper we present",
    "a hybrid pipeline that combines fast native text extraction with",
    "neural layout analysis and optical character recognition for",
    "scanned material. We evaluate the approach on a diverse corpus",
    "of born-digital and rasterised documents and report favourable",
    "results across all routing modes.",
    "Equation recognition is performed by a lightweight ONNX model",
    "that transcribes cropped formula regions into LaTeX source.",
    "x squared over a squared minus y squared over b squared equals 1",
    "Table regions are converted to GitHub-flavoured markdown using",
    "a structure-recognition pass, while figures are staged as assets",
    "for downstream embedding and retrieval pipelines.",
]

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
]


def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_scanned_page(dpi: int = 150) -> Image.Image:
    """Draw a raster page that looks like a 150-dpi scan of a text page."""
    w, h = int(8.5 * dpi), int(11 * dpi)  # US Letter
    img = Image.new("L", (w, h), 250)  # slightly off-white paper
    draw = ImageDraw.Draw(img)

    font = _find_font(int(0.24 * dpi))
    title_font = _find_font(int(0.30 * dpi))

    margin = int(0.9 * dpi)
    line_h = int(0.30 * dpi)
    y = margin

    draw.text((margin, y), "An Example Scanned Page", font=title_font, fill=30)
    y += int(0.5 * dpi)

    for para in _PARAGRAPHS:
        for line in (para[i : i + 88] for i in range(0, len(para), 88)):
            draw.text((margin, y), line, font=font, fill=40)
            y += line_h
        y += line_h

    # simulate scan noise + slight skew artefacts (uniform, deterministic)
    import random

    rng = random.Random(42)
    for _ in range(w * h // 4000):
        x, y0 = rng.randrange(w), rng.randrange(h)
        draw.point((x, y0), fill=rng.randrange(200, 250))
    return img


def generate() -> None:
    img = _draw_scanned_page()
    FIXTURES.mkdir(parents=True, exist_ok=True)

    # reportlab embeds the raster as the only page content → no text layer.
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(OUT), pagesize=(8.5 * 72, 11 * 72))
    c.drawImage(ImageReader(img), 0, 0, width=8.5 * 72, height=11 * 72, preserveAspectRatio=True)
    c.showPage()
    c.save()
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    generate()
