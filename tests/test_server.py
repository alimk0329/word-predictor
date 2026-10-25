import http.client
import importlib.util
import json
import statistics
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from test_suggest import _model

spec = importlib.util.spec_from_file_location("server", Path(__file__).parents[1] / "dashboard" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def _serve():
    server.Handler.model = _model()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _post(conn, body):
    conn.request("POST", "/predict", body)
    r = conn.getresponse()
    return r.status, r.read()


def test_predict_and_input_validation():
    httpd = _serve()
    port = httpd.server_address[1]
    try:
        status, body = _post(http.client.HTTPConnection("127.0.0.1", port), json.dumps({"text": "rocket launch s"}))
        assert status == 200 and json.loads(body)["words"][0][0] == "shuttle"
        for bad, code in [("{nope", 400), ('{"q": 1}', 400), ('{"text": 5}', 400), ("[1]", 400),
                          (json.dumps({"text": "a" * 200_000}), 413)]:
            assert _post(http.client.HTTPConnection("127.0.0.1", port), bad)[0] == code, bad
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/")
        assert conn.getresponse().status == 200
    finally:
        httpd.shutdown()


def test_keystroke_round_trip_on_one_connection_is_fast():
    httpd = _serve()
    try:
        conn, times = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1]), []
        for i in range(1, 30):
            t = time.perf_counter()
            assert _post(conn, json.dumps({"text": "the shuttle reached orbit"[:i]}))[0] == 200
            times.append(time.perf_counter() - t)
        assert statistics.median(times) < 0.02  # Nagle + delayed ACK made this ~0.04 s per keystroke
    finally:
        httpd.shutdown()
