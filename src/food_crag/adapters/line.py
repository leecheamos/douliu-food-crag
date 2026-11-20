"""LINE Messaging API: signature check, event deduplication and replies.

The webhook answers LINE immediately after the signature check; the slow
recommendation (Gemini may take tens of seconds) runs in a worker and is sent
with the event's reply token. Push messages are never used, so a reply that
misses the reply-token window is logged as a code, not retried as a push.
"""
import base64
import hashlib
import hmac
import http.client
import json
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = "https://api.line.me/v2/bot"
MAX_MESSAGES = 5          # LINE reply limit per request.
MAX_TEXT = 5000           # LINE text message limit (characters).
USER_PREFIX = "line:"     # Keeps LINE users apart from /api/chat user IDs.
HELP_TEXT = "目前只看得懂文字訊息喔！告訴我想吃什麼、預算多少，例如「想吃麵，100 元以內」。"


class LineError(RuntimeError):
    """Sanitized codes only, never tokens or provider bodies."""


def sign(secret: str, body: bytes) -> str:
    return base64.b64encode(hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()).decode("ascii")


def verify_signature(secret: str, body: bytes, signature: str | None) -> bool:
    if not secret or not signature:
        return False
    return hmac.compare_digest(sign(secret, body), signature)


def split_text(text: str, limit: int = MAX_TEXT, max_messages: int = MAX_MESSAGES) -> list[str]:
    """Split on line breaks into at most max_messages chunks of at most limit characters."""
    chunks, current = [], ""
    for line in text.strip().split("\n"):
        while len(line) > limit:  # A single overlong line is hard-wrapped.
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    if len(chunks) > max_messages:
        chunks = chunks[:max_messages]
        chunks[-1] = chunks[-1][:limit - 2] + "…"
    return chunks or [""]


class LineMessagingClient:
    def __init__(self, access_token: str, timeout: float = 10.0):
        self.access_token = access_token
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> None:
        if not self.access_token:
            raise LineError("not_configured")
        request = Request(f"{API}{path}", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {self.access_token}"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response.read(65536)
        except HTTPError as error:
            code = error.code
            error.close()
            raise LineError(f"http_{code}") from None
        except (TimeoutError, URLError, OSError, http.client.HTTPException):
            raise LineError("network_or_timeout") from None

    def reply(self, reply_token: str, texts: list[str]) -> None:
        self._post("/message/reply", {"replyToken": reply_token,
                                      "messages": [{"type": "text", "text": t} for t in texts[:MAX_MESSAGES]]})

    def show_loading(self, user_id: str, seconds: int = 60) -> None:
        """Typing animation in one-to-one chats; disappears when the reply arrives."""
        self._post("/chat/loading/start", {"chatId": user_id, "loadingSeconds": seconds})


class EventDeduplicator:
    """Remembers recent webhookEventIds so LINE redeliveries are handled once."""

    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = Lock()

    def first_time(self, event_id: str) -> bool:
        with self._lock:
            if event_id in self._seen:
                return False
            self._seen[event_id] = None
            if len(self._seen) > self.capacity:
                self._seen.popitem(last=False)
            return True


def _log(code: str) -> None:
    # Codes only: no message text, user IDs, reply tokens or credentials.
    print(f"[line] {code}", file=sys.stderr, flush=True)


class LineWebhookHandler:
    def __init__(self, pipeline, channel_secret: str, client, executor=None, dedup=None):
        self.pipeline = pipeline
        self.channel_secret = channel_secret
        self.client = client
        self.executor = executor if executor is not None else ThreadPoolExecutor(max_workers=4,
                                                                                 thread_name_prefix="line")
        self.dedup = dedup if dedup is not None else EventDeduplicator()

    @property
    def configured(self) -> bool:
        return bool(self.channel_secret and self.client.access_token)

    def accept(self, body: bytes, signature: str | None) -> int:
        """Return the HTTP status for LINE; queue valid text events for processing."""
        if not verify_signature(self.channel_secret, body, signature):
            return 401
        try:
            payload = json.loads(body)
            events = payload["events"]
            if not isinstance(events, list):
                raise TypeError
        except (ValueError, UnicodeError, KeyError, TypeError):
            return 400
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = event.get("webhookEventId")
            if isinstance(event_id, str) and event_id and not self.dedup.first_time(event_id):
                _log("duplicate_event_skipped")
                continue
            self.executor.submit(self._process, event)
        return 200

    def _process(self, event: dict) -> None:
        try:
            if event.get("type") != "message" or event.get("mode", "active") != "active":
                return  # follow/unfollow etc.: LINE's own greeting covers follow.
            reply_token = event.get("replyToken")
            user_id = (event.get("source") or {}).get("userId")
            message = event.get("message") or {}
            if not isinstance(reply_token, str) or not reply_token or not isinstance(user_id, str) or not user_id:
                return
            if message.get("type") != "text" or not isinstance(message.get("text"), str) or not message["text"].strip():
                self.client.reply(reply_token, [HELP_TEXT])
                return
            if (event.get("source") or {}).get("type") == "user":
                try:
                    self.client.show_loading(user_id)
                except LineError as error:
                    _log(f"loading_failed:{error}")
            result = self.pipeline.handle(USER_PREFIX + user_id, message["text"])
            self.client.reply(reply_token, split_text(result.reply))
            _log(f"replied:{result.status}")
        except LineError as error:
            _log(f"reply_failed:{error}")
        except Exception as error:  # A worker must never die silently.
            _log(f"processing_failed:{type(error).__name__}")
