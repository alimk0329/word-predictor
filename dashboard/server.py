"""Live topic + word suggestions.

    python dashboard/server.py [port]    # then open http://127.0.0.1:8000
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tc.suggest import Suggester, load_20ng, split, train

MODEL = Path(__file__).resolve().parents[1] / "model.pkl"
PAGE = (Path(__file__).parent / "index.html").read_bytes()
MAX_BODY = 100_000  # bytes of JSON per request


def load_model():
    if MODEL.exists():
        return Suggester.load(MODEL)
    print("No model.pkl yet: training on 20 Newsgroups (a few seconds; benchmarks/compare.py also writes it)")
    texts, y, labels, article = load_20ng("train")
    tr, va = split(y, article)
    model = train([texts[i] for i in tr], y[tr], labels, [texts[i] for i in va], y[va])
    model.save(MODEL)
    return model


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: one connection carries a whole typing session
    disable_nagle_algorithm = True  # headers and body go out as two writes; Nagle + delayed ACK adds ~40 ms
    model = None

    def do_GET(self):
        if self.path != "/":
            return self.send_error(404)
        self._send(PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/predict":
            return self.send_error(404)
        try:
            n = int(self.headers.get("Content-Length", ""))
            if not 0 <= n <= MAX_BODY:
                return self.send_error(413)  # send_error closes the connection, so the unread body is dropped
            text = json.loads(self.rfile.read(n))["text"]
            if not isinstance(text, str):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            return self.send_error(400, "expected JSON {\"text\": string}")
        self._send(json.dumps(self.model.suggest(text)).encode(), "application/json")

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # a log line per keystroke is noise
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    Handler.model = load_model()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Dashboard: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
