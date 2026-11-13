"""The release matrix must match what bobine claims to support.

`pyproject.toml` says `requires-python`; `.github/workflows/release.yml`
decides which wheels actually ship. Those two drifted once already (the
matrix comment still advertised musllinux long after it was cut), and a
missing leg is invisible until someone on that platform falls back to the
sdist and needs a Rust toolchain.

The expected table is written out on purpose: changing the matrix should be
a decision that includes updating this test, the workflow comment, and the
release notes.
"""
from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"

#: (runner, target) — the platforms bobine ships wheels for.
EXPECTED_PLATFORMS = {
    ("ubuntu-22.04", "x86_64"),
    ("ubuntu-22.04-arm", "aarch64"),
    ("windows-latest", "x64"),
    ("macos-latest", "aarch64"),
}

#: 20 wheels for 0.9.x (cp310-cp314 x 4 platforms) + sdist.
EXPECTED_PYTHON = ["3.10", "3.11", "3.12", "3.13", "3.14"]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _matrix() -> tuple[list[str], set[tuple[str, str]]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    versions = re.search(r'python-version:\s*\[([^\]]+)\]', text)
    assert versions, "no python-version list in release.yml"
    py = [v.strip().strip('"') for v in versions.group(1).split(",")]
    legs = set(re.findall(r"runner:\s*(\S+)\s*\n\s*target:\s*(\S+)", text))
    return py, legs


class MatrixMatchesTheClaim(unittest.TestCase):
    def test_python_floor_is_the_lowest_wheel(self):
        floor = _pyproject()["project"]["requires-python"]
        lowest = tuple(int(x) for x in re.findall(r"\d+", floor)[:2])
        py, _ = _matrix()
        built = tuple(int(x) for x in py[0].split(".")[:2])
        self.assertEqual(
            built,
            lowest,
            f"requires-python {floor} but the matrix starts at {py[0]}",
        )

    def test_every_claimed_python_is_built(self):
        py, _ = _matrix()
        self.assertEqual(py, EXPECTED_PYTHON, "matrix changed: update this table and the workflow comment")

    def test_platforms_are_the_documented_set(self):
        _, legs = _matrix()
        self.assertEqual(
            legs,
            EXPECTED_PLATFORMS,
            "platform coverage changed: update this table, the workflow comment "
            "and the release notes",
        )

    def test_wheel_count_matches_the_comment(self):
        py, legs = _matrix()
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("= 20 wheels + sdist", text)
        self.assertEqual(len(py) * len(legs), 20)

    def test_no_musl_leg_without_the_fix(self):
        """musllinux was cut for a proven reason; re-adding it needs surgery."""
        _, legs = _matrix()
        self.assertFalse(
            any("musl" in target for _, target in legs),
            "musllinux needs the before-script fix documented in the workflow",
        )


if __name__ == "__main__":
    unittest.main()
