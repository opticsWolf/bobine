"""The site must state the version that ships (no drift on a bump).

The release checklist bumps `Cargo.toml`, `pyproject.toml` and
`pages/index.html`; this keeps the third one honest, the way okfgraph's
`tests/test_pages.py` does.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def version_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise AssertionError(f"no version in {path}")
    return match.group(1)


class SiteMatchesTheRelease(unittest.TestCase):
    def setUp(self):
        self.version = version_of(ROOT / "Cargo.toml")
        self.index = (ROOT / "pages" / "index.html").read_text(encoding="utf-8")

    def test_pyproject_and_cargo_agree(self):
        self.assertEqual(version_of(ROOT / "pyproject.toml"), self.version)

    def test_badge_shows_the_version(self):
        self.assertIn(
            f'bobine <span class="release">{self.version}</span>', self.index
        )

    def test_published_stack_row_shows_the_version(self):
        row = re.search(
            r"<tr><th scope=\"row\">PDF / Office conversion</th>.*?</tr>",
            self.index,
        )
        self.assertIsNotNone(row, "published-stack row missing")
        self.assertIn(f"<td>{self.version}</td>", row.group(0))

    def test_stack_pin_comment_shows_the_version(self):
        self.assertIn(f"# bobine {self.version} ↔ embroider", self.index)


if __name__ == "__main__":
    unittest.main()
