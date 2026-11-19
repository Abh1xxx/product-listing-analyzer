"""HTTP interface for Vercel; uses the same analysis as the Streamlit app."""

from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.keyword_extraction import CorpusKeywordModel
from src.listing_analysis import analyze_listing, corpus_texts


@lru_cache(maxsize=1)
def dataset():
    return load_dataset(ROOT / "data" / "Product_Listing_Analyzer_500_Corpus_Detailed.xlsx")


@lru_cache(maxsize=1)
def keyword_model():
    frame, columns = dataset()
    return CorpusKeywordModel.fit(corpus_texts(frame, columns))


class handler(BaseHTTPRequestHandler):
    def route(self):
        url = urlsplit(self.path)
        path = url.path.rstrip("/") or "/"
        if path in {"/api", "/api/index"}:
            operation = parse_qs(url.query).get("operation", [""])[0]
            return f"/api/{operation}" if operation else path
        return path

    def respond(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.route()
        if path == "/api/health":
            return self.respond({"status": "ok"})
        if path == "/api/dataset":
            frame, columns = dataset()
            return self.respond({
                "records": len(frame),
                "columns": list(frame.columns),
                "fields": columns,
                "types": {name: str(dtype) for name, dtype in frame.dtypes.items()},
                "preview": json.loads(frame.head(5).to_json(orient="records")),
                "duplicates_removed": int(frame.attrs.get("duplicate_rows_removed", 0)),
            })
        # Static files are served by Vercel's CDN in production and here locally.
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/assets/app.js": ("assets/app.js", "text/javascript; charset=utf-8"),
            "/assets/style.css": ("assets/style.css", "text/css; charset=utf-8"),
        }
        if path not in files:
            return self.respond({"error": "Page not found."}, 404)
        filename, content_type = files[path]
        body = (ROOT / "public" / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.route() != "/api/analyze":
            return self.respond({"error": "Endpoint not found."}, 404)
        if self.headers.get_content_type() != "application/json":
            return self.respond({"error": "Send JSON input."}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.respond({"error": "Invalid request length."}, 400)
        if length <= 0 or length > 30000:
            return self.respond({"error": "Request must contain at most 30,000 bytes."}, 413)
        try:
            value = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            return self.respond({"error": "Invalid JSON input."}, 400)
        if not isinstance(value, dict):
            return self.respond({"error": "Send a title and description."}, 400)
        title, description = value.get("title"), value.get("description")
        if not isinstance(title, str) or not isinstance(description, str):
            return self.respond({"error": "Title and description must be text."}, 400)
        title, description = title.strip(), description.strip()
        if not title or not description or len(title) > 250 or len(description) > 5000:
            return self.respond({"error": "Enter a title (1–250 characters) and description (1–5,000 characters)."}, 400)
        try:
            frame, columns = dataset()
            result = analyze_listing(title, description, frame, columns, keyword_model())
        except LookupError:
            return self.respond({"error": "The language resources are unavailable. Please try again later."}, 503)
        except Exception:
            logging.exception("Listing analysis failed")
            return self.respond({"error": "Analysis is temporarily unavailable. Please try again."}, 500)
        return self.respond(result)


if __name__ == "__main__":
    print("Product Listing Analyzer: http://127.0.0.1:8000", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8000), handler).serve_forever()
