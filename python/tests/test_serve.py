"""The host (bobine 0.9, plan B4): python -m bobine.serve.

The conversion work is native; these tests drive the pool and the HTTP
shell with a stub converter so they need no models and no downloads.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from bobine import serve as S


class Stub:
    """A converter that records construction, use and release."""

    instances: list = []

    def __init__(self, config, cache_dir):
        self.config = config
        self.cache_dir = cache_dir
        self.closed = False
        self.conversions = 0
        Stub.instances.append(self)

    def describe(self):
        return {"device_requested": self.config.device,
                "precision_requested": self.config.precision,
                "passes": {}}

    def close(self):
        self.closed = True

    def ingest_document(self, path, output_dir, on_page=None, **_kw):
        self.conversions += 1
        for i in range(2):
            if on_page:
                on_page(i, 2)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        md = out / "doc.md"
        md.write_text("# converted\n", encoding="utf-8")
        return type("Doc", (), {
            "md_path": str(md), "image_dir": str(out / "_assets"),
            "page_count": 2, "image_count": 0,
        })()


class Failing(Stub):
    def ingest_document(self, path, output_dir, **_kw):
        raise RuntimeError("pdf_oxide error: malformed xref table")


def running(pool):
    handler = type("BoundHandler", (S.Handler,), {"pool": pool})
    httpd = S.Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, "http://127.0.0.1:%d" % httpd.server_address[1]


def post(url, payload):
    req = urllib.request.Request(
        url + "/v1/convert", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        lines = [json.loads(l) for l in resp.read().decode().splitlines() if l]
        return resp.status, lines


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=60) as resp:
        return json.loads(resp.read().decode())


class PoolRules(unittest.TestCase):
    def setUp(self):
        Stub.instances.clear()

    def test_one_converter_per_settings_key(self):
        pool = S.Pool(cache_dir="x", idle_seconds=300, converter_factory=Stub)
        first = pool.acquire({"device": "cpu"})
        again = pool.acquire({"device": "cpu"})
        self.assertIs(first, again)
        self.assertEqual(len(Stub.instances), 1)

        other = pool.acquire({"device": "cuda"})
        self.assertIsNot(other, first)
        self.assertEqual(len(Stub.instances), 2)
        # A key ignores unknown settings and None values, so a request that
        # means the same thing reuses the converter.
        pool.acquire({"device": "cpu", "bogus": 1, "precision": None})
        self.assertEqual(len(Stub.instances), 2)

    def test_idle_zero_releases_after_every_call(self):
        pool = S.Pool(cache_dir="x", idle_seconds=0, converter_factory=Stub)
        warm = pool.acquire({})
        self.assertEqual(len(pool.release_idle()), 1)
        self.assertTrue(warm.converter.closed)
        self.assertEqual(pool.report(), [])

    def test_idle_keeps_recent_converters(self):
        clock = [0.0]
        pool = S.Pool(cache_dir="x", idle_seconds=10,
                      converter_factory=Stub, clock=lambda: clock[0])
        warm = pool.acquire({})
        clock[0] = 5.0
        self.assertEqual(pool.release_idle(), [])
        self.assertFalse(warm.converter.closed)
        clock[0] = 11.0
        self.assertEqual(len(pool.release_idle()), 1)
        self.assertTrue(warm.converter.closed)

    def test_close_releases_everything(self):
        pool = S.Pool(cache_dir="x", idle_seconds=300, converter_factory=Stub)
        warm = pool.acquire({})
        pool.close()
        self.assertTrue(warm.converter.closed)
        with self.assertRaises(RuntimeError):
            pool.acquire({})


class Routes(unittest.TestCase):
    def setUp(self):
        Stub.instances.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pool = S.Pool(cache_dir=str(Path(self.tmp.name) / "cache"),
                           idle_seconds=300, converter_factory=Stub)
        self.httpd, self.url = running(self.pool)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def test_health(self):
        self.assertEqual(get(self.url, "/health")["status"], "ok")

    def test_models_lists_warm_keys(self):
        post(self.url, {"path": "a.pdf", "output_dir": self.tmp.name})
        report = get(self.url, "/v1/models")
        warm = report["data"][1]
        self.assertEqual(warm["object"], "warm")
        self.assertEqual(len(warm["data"]), 1)
        self.assertEqual(warm["data"][0]["documents"], 1)
        self.assertIn("last_used", warm["data"][0])
        self.assertIn("describe", warm["data"][0])

    def test_convert_streams_progress_then_the_result(self):
        status, lines = post(self.url, {
            "path": "a.pdf", "output_dir": self.tmp.name,
            "settings": {"device": "cpu"},
        })
        self.assertEqual(status, 200)
        self.assertEqual([l.get("page") for l in lines[:-1]], [0, 1])
        result = lines[-1]
        self.assertTrue(result["document"])
        self.assertEqual(result["page_count"], 2)
        self.assertTrue(Path(result["md_path"]).is_file())
        self.assertEqual(result["describe"]["device_requested"], "cpu")

    def test_two_calls_share_one_converter(self):
        for i in range(2):
            post(self.url, {"path": "%d.pdf" % i, "output_dir": self.tmp.name,
                            "settings": {"device": "cpu"}})
        self.assertEqual(len(Stub.instances), 1)
        warm = self.pool.report()[0]
        self.assertEqual(warm["documents"], 2)

    def test_different_settings_build_a_second_converter(self):
        post(self.url, {"path": "a.pdf", "output_dir": self.tmp.name,
                        "settings": {"device": "cpu"}})
        post(self.url, {"path": "b.pdf", "output_dir": self.tmp.name,
                        "settings": {"device": "cuda"}})
        self.assertEqual(len(Stub.instances), 2)

    def test_bad_settings_are_400(self):
        with self.assertRaises(urllib.error.HTTPError) as err:
            post(self.url, {"path": "a.pdf", "output_dir": self.tmp.name,
                            "settings": {"device": "gpu2"}})
        self.assertEqual(err.exception.code, 400)

    def test_a_missing_field_is_400(self):
        with self.assertRaises(urllib.error.HTTPError) as err:
            post(self.url, {"output_dir": self.tmp.name})
        self.assertEqual(err.exception.code, 400)

    def test_a_conversion_failure_is_a_streamed_error_line(self):
        pool = S.Pool(cache_dir=str(Path(self.tmp.name) / "c2"),
                      idle_seconds=300, converter_factory=Failing)
        httpd, url = running(pool)
        try:
            status, lines = post(url, {"path": "broken.pdf",
                                       "output_dir": self.tmp.name})
            self.assertEqual(status, 200)
            self.assertIn("malformed xref", lines[-1]["error"])
            self.assertEqual(lines[-1]["path"], "broken.pdf")
        finally:
            httpd.server_close()
            httpd.shutdown()


class RealConversion(unittest.TestCase):
    """One real (model-free) conversion through the host."""

    def test_text_file_through_the_host(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            source = tmp / "doc.txt"
            source.write_text("# Hello\n\nbody\n", encoding="utf-8")
            pool = S.Pool(cache_dir=str(tmp / "cache"), idle_seconds=300)
            httpd, url = running(pool)
            try:
                status, lines = post(url, {"path": str(source),
                                           "output_dir": str(tmp / "out")})
                self.assertEqual(status, 200)
                result = lines[-1]
                self.assertTrue(result["document"])
                self.assertTrue(Path(result["md_path"]).is_file())
                self.assertFalse(
                    any(p["landed"] for p in result["describe"]["passes"].values())
                )
            finally:
                httpd.server_close()
                httpd.shutdown()
                pool.close()


if __name__ == "__main__":
    unittest.main()
