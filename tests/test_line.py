"""Step 8: LINE webhook. Real local HTTP; the LINE API itself is mocked here."""
import json
import threading
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from food_crag.adapters.line import (HELP_TEXT, MAX_MESSAGES, MAX_TEXT, EventDeduplicator, LineError,
                                     LineMessagingClient, LineWebhookHandler, sign, split_text,
                                     verify_signature)
from food_crag.domain import ChatResult
from food_crag.pipeline import RecommendationPipeline
from food_crag.server import create_server

SECRET = "test-channel-secret"


class InlineExecutor:
    """Runs submitted work immediately so tests can assert on its effects."""
    def submit(self, fn, *args):
        fn(*args)


class FakeClient:
    access_token = "test-token"

    def __init__(self, fail_reply=False, fail_loading=False):
        self.replies, self.loading = [], []
        self.fail_reply, self.fail_loading = fail_reply, fail_loading

    def reply(self, token, texts):
        if self.fail_reply:
            raise LineError("http_400")
        self.replies.append((token, texts))

    def show_loading(self, user_id, seconds=60):
        if self.fail_loading:
            raise LineError("http_429")
        self.loading.append(user_id)


class FakePipeline:
    def __init__(self, reply="已記錄你的條件。", error=None):
        self.calls, self.reply, self.error = [], reply, error

    def handle(self, user_id, message):
        self.calls.append((user_id, message))
        if self.error:
            raise self.error
        return ChatResult(status="criteria_updated", reply=self.reply)


def text_event(text="想吃麵", user="U1", event_id="E1", token="R1", **extra):
    event = dict(type="message", mode="active", webhookEventId=event_id, replyToken=token,
                 source={"type": "user", "userId": user}, message={"type": "text", "id": "m1", "text": text})
    event.update(extra)
    return event


def body(*events):
    return json.dumps({"destination": "Ubot", "events": list(events)}).encode()


class SignatureAndSplitTests(unittest.TestCase):
    def test_signature_matches_line_definition(self):
        payload = body(text_event())
        self.assertTrue(verify_signature(SECRET, payload, sign(SECRET, payload)))
        for bad in (None, "", sign("other-secret", payload), sign(SECRET, payload + b" ")):
            self.assertFalse(verify_signature(SECRET, payload, bad))
        self.assertFalse(verify_signature("", payload, sign("", payload)))

    def test_split_respects_line_limits(self):
        self.assertEqual(split_text("a\nb"), ["a\nb"])
        long = "\n".join("x" * 3000 for _ in range(8))
        chunks = split_text(long)
        self.assertLessEqual(len(chunks), MAX_MESSAGES)
        self.assertTrue(all(len(c) <= MAX_TEXT for c in chunks))
        self.assertTrue(chunks[-1].endswith("…"))
        self.assertTrue(all(len(c) <= MAX_TEXT for c in split_text("y" * 12000)))


class WebhookHandlerTests(unittest.TestCase):
    def setUp(self):
        self.pipeline, self.client = FakePipeline(), FakeClient()
        self.line = LineWebhookHandler(self.pipeline, SECRET, self.client, InlineExecutor())

    def send(self, payload, signature=None):
        return self.line.accept(payload, sign(SECRET, payload) if signature is None else signature)

    def test_text_message_is_answered_with_reply_token_and_isolated_user_id(self):
        self.assertEqual(self.send(body(text_event())), 200)
        self.assertEqual(self.pipeline.calls, [("line:U1", "想吃麵")])
        self.assertEqual(self.client.replies, [("R1", ["已記錄你的條件。"])])
        self.assertEqual(self.client.loading, ["U1"])

    def test_invalid_or_missing_signature_is_rejected_without_processing(self):
        payload = body(text_event())
        for signature in ("", "bad", sign("other", payload)):
            self.assertEqual(self.send(payload, signature), 401)
        self.assertEqual(self.pipeline.calls, [])
        self.assertEqual(self.client.replies, [])

    def test_redelivered_event_is_processed_once(self):
        payload = body(text_event())
        self.assertEqual(self.send(payload), 200)
        redelivery = body(text_event(deliveryContext={"isRedelivery": True}))
        self.assertEqual(self.send(redelivery), 200)  # LINE must still get 200.
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertEqual(len(self.client.replies), 1)

    def test_verify_button_payload_and_bad_json(self):
        self.assertEqual(self.send(body()), 200)
        for bad in (b"not json", b'{"events": {}}', b"{}"):
            self.assertEqual(self.send(bad), 400)
        self.assertEqual(self.pipeline.calls, [])

    def test_non_text_messages_get_help_and_other_events_are_ignored(self):
        sticker = text_event(event_id="E2", token="R2")
        sticker["message"] = {"type": "sticker", "id": "m2"}
        follow = {"type": "follow", "webhookEventId": "E3", "replyToken": "R3", "source": {"type": "user", "userId": "U1"}}
        standby = text_event(event_id="E4", token="R4", mode="standby")
        self.assertEqual(self.send(body(sticker, follow, standby)), 200)
        self.assertEqual(self.client.replies, [("R2", [HELP_TEXT])])
        self.assertEqual(self.pipeline.calls, [])

    def test_failures_are_contained(self):
        failing = LineWebhookHandler(FakePipeline(error=RuntimeError("boom")), SECRET, FakeClient(), InlineExecutor())
        payload = body(text_event())
        self.assertEqual(failing.accept(payload, sign(SECRET, payload)), 200)
        client = FakeClient(fail_reply=True, fail_loading=True)
        handler = LineWebhookHandler(FakePipeline(), SECRET, client, InlineExecutor())
        self.assertEqual(handler.accept(payload, sign(SECRET, payload)), 200)

    def test_deduplicator_is_bounded(self):
        dedup = EventDeduplicator(capacity=2)
        self.assertTrue(all(dedup.first_time(i) for i in ("a", "b", "c")))
        self.assertTrue(dedup.first_time("a"))  # Oldest entry was evicted.
        self.assertFalse(dedup.first_time("c"))


class MessagingClientTests(unittest.TestCase):
    def test_reply_request_shape_and_sanitized_errors(self):
        client = LineMessagingClient("secret-token")

        class Response(BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): self.close()

        with patch("food_crag.adapters.line.urlopen", return_value=Response(b"{}")) as call:
            client.reply("R1", ["a", "b"])
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.line.me/v2/bot/message/reply")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertEqual(json.loads(request.data), {"replyToken": "R1", "messages": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
        error = HTTPError("u", 401, "x", {}, BytesIO(b"secret-token leaked?"))
        for side_effect, code in ((error, "http_401"), (TimeoutError(), "network_or_timeout"),
                                  (ConnectionResetError(), "network_or_timeout")):
            with patch("food_crag.adapters.line.urlopen", side_effect=side_effect), \
                    self.assertRaises(LineError) as raised:
                client.reply("R1", ["a"])
            self.assertEqual(str(raised.exception), code)
        with self.assertRaises(LineError):
            LineMessagingClient("").reply("R1", ["a"])


class LineHTTPTests(unittest.TestCase):
    """Real local server and the real pipeline (Gemini not configured in tests)."""

    def setUp(self):
        self.client = FakeClient()
        pipeline = RecommendationPipeline()
        self.line = LineWebhookHandler(pipeline, SECRET, self.client, InlineExecutor())
        self.server = create_server(port=0, pipeline=pipeline, line=self.line)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def post(self, payload, signature):
        request = Request(f"http://127.0.0.1:{self.server.server_port}/webhooks/line", data=payload,
                          headers={"Content-Type": "application/json", "X-Line-Signature": signature})
        try:
            with urlopen(request, timeout=10) as response:
                return response.status
        except HTTPError as error:
            error.close()
            return error.code

    def test_signed_conversation_keeps_per_user_state_and_bad_signature_is_401(self):
        steps = [("想吃麵，不要火鍋", "UA", "E1"), ("想吃火鍋", "UB", "E2"), ("100元以內", "UA", "E3")]
        for text, user, event_id in steps:
            payload = body(text_event(text, user, event_id, token=f"T-{event_id}"))
            self.assertEqual(self.post(payload, sign(SECRET, payload)), 200)
        replies = dict(self.client.replies)
        self.assertIn("料理 麵食；排除 火鍋", replies["T-E3"][0])
        self.assertIn("預算 100 元以內", replies["T-E3"][0])
        self.assertIn("料理 火鍋；排除 無", replies["T-E2"][0])
        payload = body(text_event("想吃飯", "UA", "E4"))
        self.assertEqual(self.post(payload, "forged"), 401)
        self.assertEqual(len(self.client.replies), 3)
        with urlopen(f"http://127.0.0.1:{self.server.server_port}/health", timeout=5) as response:
            self.assertTrue(json.load(response)["line_configured"])

    def test_chat_api_can_be_disabled_for_public_exposure(self):
        server = create_server(port=0, pipeline=FakePipeline(), line=self.line, chat_api=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(f"http://127.0.0.1:{server.server_port}/api/chat",
                              data=json.dumps({"user_id": "x", "message": "想吃麵"}).encode(),
                              headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=5)
            self.assertEqual(raised.exception.code, 404)
            raised.exception.close()
            payload = body(text_event(event_id="E9"))
            request = Request(f"http://127.0.0.1:{server.server_port}/webhooks/line", data=payload,
                              headers={"Content-Type": "application/json", "X-Line-Signature": sign(SECRET, payload)})
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main()
