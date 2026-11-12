"""Runtime discovery (bobine 0.9, plan B2).

Run with the standard library — no pytest, no new dev dependency::

    python -m unittest discover -s python/tests -v

The import-hygiene test runs in a **subprocess** on purpose: the process
running the tests may itself have imported ``onnxruntime``, and the claim
under test is about a *fresh* process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run_python(code: str, env: dict | None = None,
               unset: tuple[str, ...] = ()) -> dict:
    """Run *code* in a fresh interpreter and return its JSON answer."""
    full_env = dict(os.environ)
    for key in unset:
        full_env.pop(key, None)
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=180,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


class ImportHygiene(unittest.TestCase):
    """`import bobine` must not import `onnxruntime`."""

    def test_import_does_not_import_onnxruntime(self):
        out = run_python(
            """
            import json, sys
            import bobine
            print(json.dumps({
                "ort_imported": "onnxruntime" in sys.modules,
                "dylib": bobine.ORT_DYLIB_PATH,
            }))
            """
        )
        self.assertFalse(
            out["ort_imported"],
            "importing bobine imported onnxruntime — the locator must read "
            "dist metadata, not the module",
        )

    def test_the_locator_still_finds_a_runtime(self):
        """Hygiene must not be bought with a broken discovery."""
        # Unset the variable the rest of the suite may export: this test
        # is about *discovery*, not about an explicit setting winning.
        out = run_python(
            """
            import json, sys
            import bobine
            print(json.dumps(bobine.runtime_info()))
            """,
            unset=("ORT_DYLIB_PATH",),
        )
        if out["source"] == "os":
            self.skipTest("no onnxruntime installed in this environment")
        self.assertIsNotNone(out["dylib"])
        self.assertTrue(
            Path(out["dylib"]).is_file(),
            f"located dylib does not exist: {out['dylib']}",
        )
        self.assertTrue(out["source"].startswith("pip:"), out["source"])
        self.assertEqual(out["set_by"], "bobine")
        self.assertIsInstance(out["version"], str)


class ExplicitSetting(unittest.TestCase):
    """An explicit ORT_DYLIB_PATH always wins, and is reported as such."""

    def test_explicit_env_wins(self):
        fake = str(REPO / "definitely-not-a-real-onnxruntime.dll")
        out = run_python(
            """
            import json
            import bobine
            print(json.dumps({"dylib": bobine.ORT_DYLIB_PATH,
                              "info": bobine.runtime_info()}))
            """,
            env={"ORT_DYLIB_PATH": fake},
        )
        self.assertEqual(out["dylib"], fake)
        self.assertEqual(out["info"]["dylib"], fake)
        self.assertEqual(out["info"]["source"], "env")
        self.assertEqual(out["info"]["set_by"], "env")


class ReportShape(unittest.TestCase):
    def test_python_sets_no_policy(self):
        """The wrapper decides nothing: no env writes of its own.

        `ORT_DYLIB_PATH` comes from the native bootstrap, and importing
        bobine twice must not re-probe or overwrite an explicit setting.
        """
        import bobine

        self.assertEqual(bobine.ORT_DYLIB_PATH, bobine._RUNTIME["dylib"])

    def test_runtime_info_shape(self):
        import bobine

        info = bobine.runtime_info()
        self.assertEqual(
            set(info),
            {"dylib", "source", "version", "set_by", "cuda_dirs"},
        )
        self.assertIsInstance(info["cuda_dirs"], list)
        for d in info["cuda_dirs"]:
            self.assertIsInstance(d, str)

    def test_a_missing_runtime_is_data_not_an_error(self):
        """A machine with no ONNX Runtime still imports and reports.

        The decision lives in Rust (`src/runtime.rs`, unit-tested there);
        this checks the wrapper hands a "nothing found" answer through
        unchanged instead of raising.
        """
        import bobine

        original = bobine._native.runtime_info
        bobine._native.runtime_info = lambda: {
            "dylib": None, "source": "os", "version": None,
            "set_by": "none", "cuda_dirs": [],
        }
        try:
            info = bobine.runtime_info()
        finally:
            bobine._native.runtime_info = original
        self.assertIsNone(info["dylib"])
        self.assertEqual(info["source"], "os")
        self.assertEqual(info["set_by"], "none")


if __name__ == "__main__":
    unittest.main()
