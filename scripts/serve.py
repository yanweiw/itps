"""Tiny static file server with the cross-origin headers ORT Web's threaded
WASM backend needs (SharedArrayBuffer requires COOP/COEP).

If WebGPU is available in the visitor's browser, these headers don't matter --
ORT runs everything on the GPU. They only kick in for the WASM fallback path,
where they unlock multi-threaded SIMD WASM (~4x faster than single-threaded).

Usage (from repo root):
    python scripts/serve.py [port]   # default port 8000

Open http://localhost:8000 in Chrome / Edge / Safari 18+.
"""

from __future__ import annotations

import functools
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class CrossOriginIsolationHandler(SimpleHTTPRequestHandler):
    """Adds COOP / COEP / CORP headers required for SharedArrayBuffer."""

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        # Help dev iteration: never let the browser cache act.onnx silently.
        if self.path.endswith(".onnx") or self.path.endswith(".wasm"):
            self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    handler = functools.partial(CrossOriginIsolationHandler, directory=repo_root)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving {repo_root} at http://localhost:{port}")
    print("  COOP / COEP headers enabled -> SharedArrayBuffer / multi-threaded WASM available")
    print("  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.server_close()


if __name__ == "__main__":
    main()
