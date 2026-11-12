"""The stub is documentation: it must describe the object that exists.

`python/bobine/__init__.pyi` is the documented contract for editors and
type checkers, and it is hand-maintained, so it drifts silently. This
compares it with the imported extension: every public module name and every
public `HybridConverter` method must appear in the stub, and the stub may
not advertise a module-level name the package does not have.
"""
from __future__ import annotations

import ast
import types
import unittest
from pathlib import Path

import bobine

STUB = Path(__file__).resolve().parents[1] / "bobine" / "__init__.pyi"


def _stub_surface():
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    names, methods = set(), set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
            if node.name == "HybridConverter":
                methods |= {
                    n.name for n in node.body if isinstance(n, ast.FunctionDef)
                }
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names, methods


class StubMatchesThePackage(unittest.TestCase):
    def test_module_surface(self):
        stub, _ = _stub_surface()
        # Submodules (`serve`) are attributes once imported, but they are
        # documented by their own docstring, not by `__init__.pyi`.
        real = {
            n for n in dir(bobine)
            if not n.startswith("_")
            and not isinstance(getattr(bobine, n), types.ModuleType)
        }
        self.assertEqual(sorted(real - stub), [], "public names missing from the stub")
        self.assertEqual(sorted(stub - real), [], "stub advertises names that do not exist")

    def test_converter_methods(self):
        _, stub = _stub_surface()
        real = {n for n in dir(bobine.HybridConverter) if not n.startswith("_")}
        real |= {"__enter__", "__exit__"}
        self.assertEqual(sorted(real - stub), [], "methods missing from the stub")
        self.assertEqual(sorted(stub - real - {"__init__"}), [])

    def test_the_host_is_a_module_with_docs(self):
        from bobine import serve

        self.assertTrue(serve.__doc__ and "NDJSON" in serve.__doc__)
        self.assertTrue(callable(serve.main))

    def test_the_09_surface_is_documented(self):
        stub, methods = _stub_surface()
        for name in ("DEVICES", "PRECISIONS", "runtime_info", "describe"):
            self.assertIn(name, stub | methods)


if __name__ == "__main__":
    unittest.main()
