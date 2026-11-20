"""Real local HTTP smoke checks; no external API or LINE claims."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from food_crag.server import create_server


class ScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base + path, data=body, headers={"Content-Type": "application/json"})
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_health_reports_recommendation_ready(self):
        code, body = self.request("/health")
        self.assertEqual(code, 200)
        self.assertTrue(body["recommendation_ready"])

    def test_chat_does_not_invent_restaurants(self):
        code, body = self.request("/api/chat", {"user_id": "demo", "message": "晚餐吃什麼"})
        self.assertEqual(code, 200)
        self.assertEqual(body["criteria"]["meal"], "dinner")
        self.assertEqual(body["restaurant_ids"], [])

    def test_bad_input(self):
        self.assertEqual(self.request("/api/chat", {"message": "午餐"})[0], 400)

    def test_unconfigured_line_webhook_is_refused(self):
        self.assertEqual(self.request("/webhooks/line", {"events": []}), (503, {"error": "line_not_configured"}))


if __name__ == "__main__":
    unittest.main()
