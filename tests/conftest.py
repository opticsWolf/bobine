"""Shared fixtures and fake pdf_oxide objects for bobine tests.

The HybridConverter talks to pdf_oxide through a small duck-typed surface:

- ``PdfDocument``: context manager, iterable pages, ``len(doc)`` or
  ``doc.page_count()``, ``doc.extract_image_bytes(i)`` → list of ``{data,
  format}``, ``doc.within(i, (x, y, w, h))`` → region with ``extract_text()``.
- ``Page``: ``markdown(detect_headings=...)`` / ``plain_text()`` / ``text``,
  ``chars`` (each with ``char`` / ``font_name`` / ``bbox``), ``images``,
  ``width`` / ``height`` (points), ``index``, ``render(dpi=...)``.

Fakes let the whole routing / splicing / image-staging logic be exercised with
zero native dependencies. They are also used as the basis for the optional
integration suite when the real ``pdf_oxide`` is installed.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import pytest

# 1x1 transparent PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# ── fake pdf_oxide objects ────────────────────────────────────────────────


@dataclass
class FakeChar:
    char: str
    font_name: str = "Helvetica"
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 12.0)


@dataclass
class FakePage:
    """A page with a usable text layer, mimicking pdf_oxide's Page."""

    text: str = ""
    chars: list[FakeChar] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)
    width: float = 612.0
    height: float = 792.0
    index: int = 0
    markdown_text: str | None = None
    render_result: Any = None

    # -- pdf_oxide surface used by HybridConverter -------------------------
    def markdown(self, detect_headings: bool = True) -> str:
        if self.markdown_text is not None:
            return self.markdown_text
        return self.text

    def plain_text(self) -> str:
        return self.text

    def render(self, dpi: int = 72, **kwargs):
        return self.render_result

    def render_to_image(self, dpi: int = 72, **kwargs):
        return self.render_result

    # -- helpers for building fake pages -----------------------------------
    @classmethod
    def from_text(cls, text: str, *, font: str = "Helvetica", index: int = 0, **kw) -> FakePage:
        """Build a page whose ``chars`` mirror the text (one FakeChar per char)."""
        chars = []
        x = 50.0
        for ch in text:
            chars.append(FakeChar(ch, font_name=font, bbox=(x, 700.0, x + 8.0, 712.0)))
            x += 9.0
        return cls(text=text, chars=chars, index=index, **kw)

    @classmethod
    def with_images(cls, n: int = 1, **kw) -> FakePage:
        page = cls(text="", chars=[], **kw)
        page.images = [{"data": PNG, "format": "png"} for _ in range(n)]
        return page


@dataclass
class FakeRegion:
    text: str = ""

    def extract_text(self) -> str:
        return self.text


class FakePdfDocument:
    """Mimics pdf_oxide.PdfDocument for converter unit tests."""

    def __init__(self, pages: list[FakePage] | None = None):
        self.pages = pages or []
        self.extracted: dict[int, list[dict]] = {}
        self.region_texts: dict[int, str] = {}

    # -- context manager + iteration ---------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __len__(self) -> int:
        return len(self.pages)

    def page_count(self) -> int:
        return len(self.pages)

    def __iter__(self):
        return iter(self.pages)

    # -- pdf_oxide surface used by HybridConverter -------------------------
    def extract_image_bytes(self, index: int) -> list[dict]:
        return self.extracted.get(index, [])

    def within(self, index: int, box: tuple[float, float, float, float]):
        return FakeRegion(self.region_texts.get(index, ""))

    def render(self, index: int, dpi: int = 72, **kwargs):
        if index < len(self.pages):
            return self.pages[index].render_result
        return None


def make_page_with_math(chars: list[FakeChar], text: str, **kw) -> FakePage:
    """Build a page whose chars mix math-font and math-unicode glyphs."""
    return FakePage(text=text, chars=chars, **kw)


# ── shared fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def png_bytes() -> bytes:
    return PNG


@pytest.fixture
def fake_doc() -> FakePdfDocument:
    """A two-page document: one normal text page, one scanned (image-only)."""
    p0 = FakePage.from_text("Hello world from page one.", index=0)
    p1 = FakePage.with_images(n=1, index=1)
    return FakePdfDocument(pages=[p0, p1])
