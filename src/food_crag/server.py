"""HTTP server: /health, /api/chat (local testing) and /webhooks/line."""
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .adapters.line import LineMessagingClient, LineWebhookHandler
from .config import Settings
from .pipeline import RecommendationPipeline

MAX_BODY_BYTES = 16_384
MAX_LINE_BODY_BYTES = 1_048_576


def make_handler(pipeline=None, line=None, chat_api=True):
    pipeline = pipeline if pipeline is not None else RecommendationPipeline()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Avoid recording user messages and request paths in this scaffold.

        def respond(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ok", "mode": "gemini_evaluation", "recommendation_ready": True,
                                   "embedding": pipeline.embedding_status,
                                   "line_configured": bool(line is not None and line.configured)})
            else:
                self.respond(404, {"error": "not_found"})

        def do_POST(self):
            if self.path == "/webhooks/line":
                self.line_webhook()
                return
            if self.path != "/api/chat" or not chat_api:
                self.respond(404, {"error": "not_found"})
                return
            if self.headers.get_content_type() != "application/json":
                self.respond(415, {"error": "expected_application_json"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.respond(400, {"error": "invalid_content_length"})
                return
            if size <= 0 or size > MAX_BODY_BYTES:
                self.respond(413 if size > MAX_BODY_BYTES else 400, {"error": "invalid_body_size"})
                return
            try:
                payload = json.loads(self.rfile.read(size))
            except (ValueError, UnicodeError):
                self.respond(400, {"error": "invalid_json"})
                return
            if not isinstance(payload, dict) or any(
                not isinstance(payload.get(key), str) or not payload[key].strip()
                for key in ("user_id", "message")
            ):
                self.respond(400, {"error": "user_id_and_message_required"})
                return
            self.respond(200, asdict(pipeline.handle(payload["user_id"], payload["message"])))

        def line_webhook(self):
            if line is None or not line.configured:
                self.respond(503, {"error": "line_not_configured"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.respond(400, {"error": "invalid_content_length"})
                return
            if size <= 0 or size > MAX_LINE_BODY_BYTES:
                self.respond(413 if size > MAX_LINE_BODY_BYTES else 400, {"error": "invalid_body_size"})
                return
            # Signature is checked on the exact raw bytes before any parsing.
            code = line.accept(self.rfile.read(size), self.headers.get("X-Line-Signature"))
            self.respond(code, {} if code == 200 else
                         {"error": "invalid_signature" if code == 401 else "invalid_payload"})

    return Handler


def create_server(host="127.0.0.1", port=8000, pipeline=None, line=None, chat_api=True):
    return ThreadingHTTPServer((host, port), make_handler(pipeline, line, chat_api))


def line_from_settings(pipeline, settings):
    client = LineMessagingClient(settings.line_channel_access_token)
    return LineWebhookHandler(pipeline, settings.line_channel_secret, client)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", help="Explicit KEY=value file; process variables take precedence")
    args = parser.parse_args()
    settings = Settings.from_env(args.env_file)
    pipeline = RecommendationPipeline(settings=settings)
    line = line_from_settings(pipeline, settings)
    with create_server(settings.host, settings.port, pipeline, line, settings.chat_api_enabled) as server:
        print(f"Listening on http://{settings.host}:{settings.port} "
              f"(LINE webhook {'ready' if line.configured else 'not configured'}; "
              f"/api/chat {'on' if settings.chat_api_enabled else 'off'})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
