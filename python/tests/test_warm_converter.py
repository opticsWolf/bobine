"""The warm converter (bobine 0.9, plan B4).

A converter the caller holds keeps its ONNX sessions between documents;
`close()` releases them and the object refuses further work. Text files
need no models, so these tests are cheap and always run.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import bobine


def _text_doc(tmp: Path, name: str = "doc.txt", body: str = "# Hi\n\nbody\n") -> Path:
    path = tmp / name
    path.write_text(body, encoding="utf-8")
    return path


class IngestOnConverter(unittest.TestCase):
    def test_ingest_returns_the_staged_document(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            source = _text_doc(tmp)
            conv = bobine.HybridConverter(
                bobine.ConverterConfig(), cache_dir=str(tmp / "cache")
            )
            out = tmp / "out"
            doc = conv.ingest_document(str(source), str(out))
            self.assertTrue(Path(doc.md_path).is_file())
            self.assertIn("Hi", Path(doc.md_path).read_text(encoding="utf-8"))
            self.assertEqual(doc.page_count, 0)
            self.assertIn("_assets", doc.image_dir)

    def test_three_files_on_one_converter(self):
        """The point of B4: no per-file construction."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            conv = bobine.HybridConverter(
                bobine.ConverterConfig(), cache_dir=str(tmp / "cache")
            )
            for i in range(3):
                source = _text_doc(tmp, f"doc{i}.txt", f"# Doc {i}\n")
                doc = conv.ingest_document(str(source), str(tmp / f"out{i}"))
                self.assertTrue(Path(doc.md_path).is_file())

    def test_should_continue_stops_between_pages(self):
        """A text file has no pages, so the hook is simply never asked."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            conv = bobine.HybridConverter(
                bobine.ConverterConfig(), cache_dir=str(tmp / "cache")
            )
            asked = []
            doc = conv.ingest_document(
                str(_text_doc(tmp)),
                str(tmp / "out"),
                should_continue=lambda: asked.append(1) or False,
            )
            self.assertEqual(asked, [])
            self.assertTrue(Path(doc.md_path).is_file())

    def test_a_missing_file_is_still_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            conv = bobine.HybridConverter(
                bobine.ConverterConfig(), cache_dir=str(tmp / "cache")
            )
            with self.assertRaises(RuntimeError):
                conv.ingest_document(str(tmp / "absent.pdf"), str(tmp / "out"))


class Close(unittest.TestCase):
    def test_close_is_idempotent_and_refuses_work(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            conv = bobine.HybridConverter(
                bobine.ConverterConfig(), cache_dir=str(tmp / "cache")
            )
            self.assertFalse(conv.is_closed())
            conv.close()
            conv.close()
            self.assertTrue(conv.is_closed())
            with self.assertRaises(RuntimeError) as err:
                conv.ingest_document(str(_text_doc(tmp)), str(tmp / "out"))
            self.assertIn("closed", str(err.exception))

    def test_context_manager_closes(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with bobine.HybridConverter(
                bobine.ConverterConfig(), cache_dir=str(tmp / "cache")
            ) as conv:
                doc = conv.ingest_document(str(_text_doc(tmp)), str(tmp / "out"))
                self.assertTrue(Path(doc.md_path).is_file())
            self.assertTrue(conv.is_closed())

    def test_describe_still_answers_after_close(self):
        conv = bobine.HybridConverter(
            bobine.ConverterConfig(), cache_dir="~/.cache/bobine"
        )
        conv.close()
        report = conv.describe()
        self.assertFalse(any(p["landed"] for p in report["passes"].values()))


class ModuleFunctionUnchanged(unittest.TestCase):
    def test_the_one_shot_function_still_works(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            doc = bobine.ingest_document(
                str(_text_doc(tmp)), str(tmp / "out"), cache_dir=str(tmp / "cache")
            )
            self.assertTrue(Path(doc.md_path).is_file())


if __name__ == "__main__":
    unittest.main()
