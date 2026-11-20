"""Phase 6 v2: local web app (GUI) for the pigment recommender. Python standard library only.

Run: python -m app.web        then open http://127.0.0.1:8000
Only reachable from this computer (binds to 127.0.0.1).
"""
import argparse
import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from model.recommend import recommend_for_location

PAGE = Path(__file__).parent / "static" / "index.html"


def api_recommend(query):
    """(HTTP status, JSON body) for /api/recommend?lat=..&lon=..[&time=ISO-8601]."""
    q = {k: v[0] for k, v in parse_qs(query).items()}
    try:
        lat, lon = float(q["lat"]), float(q["lon"])
        when = pd.Timestamp(q["time"]) if q.get("time") else None
        if when is not None and when.tzinfo is None:
            raise ValueError("time must include a timezone, e.g. 2026-11-21T00:00:00+00:00")
    except (KeyError, ValueError) as e:
        return HTTPStatus.BAD_REQUEST, {"error": f"Give numeric lat and lon (and an optional ISO time). {e}"}
    try:
        return HTTPStatus.OK, recommend_for_location(lat, lon, when)
    except ValueError as e:  # outside Malaysia
        return HTTPStatus.BAD_REQUEST, {"error": str(e)}
    except requests.RequestException as e:
        return HTTPStatus.BAD_GATEWAY, {"error": f"Could not reach the Open-Meteo weather service: {e}"}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/recommend":
            status, body = api_recommend(url.query)
            self._send(status, json.dumps(body).encode(), "application/json")
        else:
            self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"DSSC pigment recommender running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
