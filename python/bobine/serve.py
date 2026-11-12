"""One warm converter behind HTTP: ``python -m bobine.serve``.

The conversion work is Rust (``bobine._native``); this module is the
process shell around it, the same shape embroider's host uses: a pool of
one ``HybridConverter`` per settings key, kept warm so a second file does
not reload the layout/OCR models, and released when idle.

======================== ===============================================
Route                    Body -> answer
======================== ===============================================
``GET /health``          ``{"status": "ok"}``
``GET /v1/models``       ``{"object": "list", "data": [...]}`` — cache
                         status plus the warm keys (settings, last use,
                         documents converted, ``describe()``)
``POST /v1/convert``     ``{"path", "output_dir", "settings"?}`` ->
                         NDJSON: one ``{"page", "total"}`` line per page,
                         then the result
======================== ===============================================

- **One converter per settings key.** The key is the canonical form of
  ``settings`` (sorted JSON, no ``None`` values), so two requests that
  mean the same thing share a converter and two that differ do not.
- **Serialised per key.** A second request for a busy key waits; the
  native converter runs one document at a time.
- **Idle eviction.** ``--idle-seconds`` (default 300) releases a warm
  converter after that long without use; ``0`` releases after every
  conversion (the pre-0.9 behaviour, kept for callers who want it).
- **Errors.** 400 for a bad request or settings, 500 for a failed
  conversion with ``{"error": "<bobine message>", "path": ...}`` — the text
  a consumer maps onto its own conversion error.

Exit codes: 2 for bad arguments, 3 when the address is taken. Once
listening it prints ``listening <url>`` on stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

if sys.path and os.path.abspath(sys.path[0] or ".") == os.path.dirname(
        os.path.abspath(__file__)):
    sys.path.pop(0)   # run as a script: its folder must not shadow stdlib

import bobine  # noqa: E402

#: Largest request body accepted, in bytes.
MAX_BODY = 1 * 1024 * 1024

#: Settings a request may pass (the rest stay at their defaults, so the
#: canonical key cannot be poisoned by typos).
SETTINGS = (
    "routing_mode", "extract_images", "device", "precision",
    "model_precision", "model_quantization", "table_backend",
    "use_onnx", "image_mode", "render_dpi", "formula_dpi",
)


def canonical_key(settings: Optional[Dict[str, Any]]) -> str:
    """The pool key: sorted JSON of the known settings, ``None`` dropped."""
    chosen = {
        k: v for k, v in (settings or {}).items()
        if k in SETTINGS and v is not None
    }
    return json.dumps(chosen, sort_keys=True, separators=(",", ":"))


class Warm:
    """One converter and when it was last used."""

    def __init__(self, settings: Dict[str, Any], converter: Any) -> None:
        self.settings = settings
        self.converter = converter
        self.lock = threading.Lock()
        self.last_used = time.monotonic()
        self.documents = 0

    def report(self) -> Dict[str, Any]:
        return {
            "settings": self.settings,
            "last_used": round(self.last_used, 3),
            "documents": self.documents,
            "describe": self.converter.describe(),
        }


class Pool:
    """One warm converter per settings key, released when idle."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        idle_seconds: float = 300.0,
        converter_factory: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_dir = cache_dir or "~/.cache/bobine"
        self.idle_seconds = max(0.0, float(idle_seconds))
        self._factory = converter_factory or bobine.HybridConverter
        self._clock = clock
        self._lock = threading.Lock()
        self._warm: Dict[str, Warm] = {}
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def acquire(self, settings: Optional[Dict[str, Any]] = None) -> Warm:
        key = canonical_key(settings)
        with self._lock:
            if self._closed:
                raise RuntimeError("pool is shut down")
            warm = self._warm.get(key)
            if warm is None:
                config = bobine.ConverterConfig(**(settings or {}))
                warm = Warm(
                    dict(settings or {}),
                    self._factory(config, cache_dir=self.cache_dir),
                )
                self._warm[key] = warm
            warm.last_used = self._clock()
            return warm

    def note_document(self, key: str) -> None:
        with self._lock:
            warm = self._warm.get(key)
            if warm is not None:
                warm.documents += 1
                warm.last_used = self._clock()

    def release_idle(self) -> List[str]:
        """Close warm converters older than ``idle_seconds``; return keys."""
        now = self._clock()
        released: List[str] = []
        with self._lock:
            for key, warm in list(self._warm.items()):
                if self.idle_seconds and now - warm.last_used < self.idle_seconds:
                    continue
                # A busy converter is skipped: its request owns it.
                if not warm.lock.acquire(blocking=False):
                    continue
                try:
                    warm.converter.close()
                    del self._warm[key]
                    released.append(key)
                finally:
                    warm.lock.release()
        return released

    def report(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [w.report() for w in self._warm.values()]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for warm in self._warm.values():
                warm.converter.close()
            self._warm.clear()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    pool: Pool  # set by serve()

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # keep stderr quiet
        sys.stderr.write("bobine.serve: " + (fmt % args) + "\n")

    def _reply(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/health":
            self._reply(200, {"status": "ok", "backend": "bobine"})
        elif self.path == "/v1/models":
            try:
                status = bobine.model_status()
            except Exception:  # offline cache lookup; failures are data
                status = []
            self._reply(200, {
                "object": "list",
                "data": [{"object": "model", "cache": status}, {"object": "warm",
                          "data": self.pool.report()}],
            })
        else:
            self._reply(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/v1/convert":
            self._reply(404, {"error": f"no route {self.path}"})
            return
        try:
            request = self._body()
            path = request["path"]
            output_dir = request["output_dir"]
            settings = request.get("settings") or {}
        except KeyError as exc:
            self._reply(400, {"error": f"missing field {exc}"})
            return
        except (ValueError, TypeError) as exc:
            self._reply(400, {"error": f"bad request body: {exc}"})
            return

        try:
            warm = self.pool.acquire(settings)
        except (ValueError, TypeError) as exc:  # bad settings words
            self._reply(400, {"error": str(exc)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        def line(payload: Dict[str, Any]) -> None:
            self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
            self.wfile.flush()

        def on_page(index: int, total: int) -> None:
            line({"page": index, "total": total})

        try:
            with warm.lock:
                doc = warm.converter.ingest_document(
                    path, output_dir, on_page=on_page
                )
                warm.last_used = self.pool._clock()
                warm.documents += 1
            line({
                "md_path": doc.md_path,
                "image_dir": doc.image_dir,
                "page_count": doc.page_count,
                "image_count": doc.image_count,
                "document": True,
                "describe": warm.converter.describe(),
            })
        except Exception as exc:  # noqa: BLE001 - the answer is the message
            line({"error": str(exc), "path": path})
        finally:
            self.close_connection = True
            self.pool.release_idle()


class Server(ThreadingHTTPServer):
    daemon_threads = True


def serve(
    host: str = "127.0.0.1",
    port: int = 8757,
    cache_dir: Optional[str] = None,
    idle_seconds: float = 300.0,
) -> None:
    pool = Pool(cache_dir=cache_dir, idle_seconds=idle_seconds)

    handler = type("BoundHandler", (Handler,), {"pool": pool})
    try:
        httpd = Server((host, port), handler)
    except OSError as exc:
        sys.stderr.write(f"bobine.serve: cannot bind {host}:{port}: {exc}\n")
        raise SystemExit(3)

    stop = threading.Event()

    def sweep() -> None:
        interval = max(0.05, min(5.0, idle_seconds / 4 or 0.05))
        while not stop.wait(interval):
            pool.release_idle()

    sweeper = threading.Thread(target=sweep, name="bobine-idle", daemon=True)
    sweeper.start()
    sys.stderr.write(f"listening http://{host}:{port}\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
        pool.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bobine.serve",
        description="One warm bobine converter behind HTTP.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8757)
    parser.add_argument("--cache-dir", default=None,
                        help="converter-model hub cache (default: standard HF cache)")
    parser.add_argument("--idle-seconds", type=float, default=300.0,
                        help="release a warm converter after this long unused; 0 = after every call")
    args = parser.parse_args(argv)
    if args.port < 0 or args.port > 65535:
        sys.stderr.write("bobine.serve: --port must be 0-65535\n")
        return 2
    serve(args.host, args.port, args.cache_dir, args.idle_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
