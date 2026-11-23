"""Local/mocked evidence; live Gemini validation is run separately and explicitly."""
from dataclasses import replace
import http.client
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from food_crag.adapters.gemini import GeminiError, GeminiLanguageModel
from food_crag.config import Settings
from food_crag.dataset import load_restaurants
from food_crag.domain import SearchCriteria
from food_crag.evaluator import GeminiEvaluator, SCHEMA, allowed_rewrites, retrieve_and_evaluate
from food_crag.pipeline import RecommendationPipeline
from food_crag.retriever import RestaurantRetriever
from food_crag.server import create_server


class FakeModel:
    def __init__(self, mode="accept"):
        self.mode = mode
        self.calls = []

    def generate_json(self, payload, schema):
        self.calls.append(payload)
        if self.mode == "error":
            raise GeminiError("http_429")
        if self.mode == "retry" or (self.mode == "retry_once" and len(self.calls) == 1):
            rewrites = payload["allowed_rewrites"]
            return dict(decision="retry", evidence_state="insufficient", accepted_ids=[],
                        rewrite_query=rewrites[0] if rewrites else "麵類", reasons=["測試用子類別擴展"])
        return dict(decision="accept", evidence_state="sufficient", rewrite_query="",
                    accepted_ids=[payload["candidates"][0]["restaurant"]["id"]], reasons=["測試用證據足夠"])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.retriever = RestaurantRetriever(load_restaurants())
        self.criteria = SearchCriteria(cuisines=("麵食",), excluded_cuisines=("火鍋",))
        self.report = self.retriever.search("麵食", self.criteria)
        self.model = FakeModel()
        self.evaluator = GeminiEvaluator(self.retriever, self.model)

    def test_accept_requires_local_and_model_agreement(self):
        evaluation = self.evaluator.evaluate(self.criteria, self.report)
        self.assertEqual(evaluation.decision, "accept")
        self.assertEqual(evaluation.accepted_ids, (self.report.candidates[0].restaurant.id,))
        self.assertTrue(evaluation.model_called)
        self.assertNotIn("user_id", self.model.calls[0])
        self.assertEqual(self.model.calls[0]["criteria"]["excluded_cuisines"], ("火鍋",))

    def test_missing_price_and_empty_results_do_not_call_model(self):
        unpriced = RestaurantRetriever([replace(r, price_min_twd=None, price_max_twd=None, price_evidence=None, menu_prices=())
                                        for r in load_restaurants()])
        for retriever, criteria, expected in [(unpriced, replace(self.criteria, budget_twd=200), "clarify"),
                                              (self.retriever, replace(self.criteria, district="虎尾鎮"), "stop")]:
            with self.subTest(expected=expected):
                report = retriever.search("麵食", criteria)
                evaluation = GeminiEvaluator(retriever, self.model).evaluate(criteria, report)
                self.assertEqual(evaluation.decision, expected)
                self.assertFalse(evaluation.model_called)
        self.assertEqual(self.model.calls, [])

    def test_forged_candidates_scores_and_missing_provenance_are_vetoed(self):
        candidate = self.report.candidates[0]
        bad_candidates = [
            replace(candidate, restaurant=replace(candidate.restaurant, id="INVENTED")),
            replace(candidate, restaurant=replace(candidate.restaurant, price_max_twd=1)),
            replace(candidate, score=float("nan")), replace(candidate, score=0),
            replace(candidate, score=candidate.score / 2), replace(candidate, evidence=()),
            replace(candidate, matched_terms=()),
            replace(candidate, matched_terms=("捏造命中詞",)),
            replace(candidate, query_coverage=.123),
        ]
        for bad in bad_candidates:
            with self.subTest(candidate_id=bad.restaurant.id):
                evaluation = self.evaluator.evaluate(self.criteria, replace(self.report, candidates=(bad,)))
                self.assertEqual(evaluation.decision, "stop")
                self.assertEqual(evaluation.evidence_state, "error")
        no_sources = replace(candidate.restaurant, sources=())
        retriever = RestaurantRetriever([no_sources])
        result = GeminiEvaluator(retriever, self.model).evaluate(SearchCriteria(), retriever.search("麵食", SearchCriteria()))
        self.assertEqual(result.evidence_state, "error")
        self.assertFalse(self.model.calls)

    def test_model_cannot_override_budget_veto(self):
        evaluation = self.evaluator.evaluate(replace(self.criteria, budget_twd=1), self.report)
        self.assertEqual(evaluation.decision, "stop")
        self.assertFalse(self.model.calls)

    def test_invalid_model_outputs_fail_closed(self):
        valid = dict(decision="accept", evidence_state="sufficient", accepted_ids=["DL-017"], reasons=["測試"], rewrite_query="")
        variants = [dict(valid, accepted_ids=["INVENTED"]), dict(valid, accepted_ids=[]),
                    dict(valid, accepted_ids=["DL-017", "DL-017"]), dict(valid, reasons=[]),
                    dict(valid, evidence_state="insufficient"), dict(valid, decision=[]),
                    dict(valid, extra="ignored"), dict(valid, rewrite_query="火鍋"),
                    dict(valid, decision="retry", evidence_state="insufficient", accepted_ids=[], rewrite_query="牛肉麵")]
        for output in variants:
            with self.subTest(output=output):
                with patch.object(self.model, "generate_json", return_value=output):
                    result = self.evaluator.evaluate(self.criteria, self.report)
                self.assertEqual(result.decision, "stop")
                self.assertEqual(result.evidence_state, "error")

    def test_retry_is_once_with_identical_hard_conditions(self):
        model = FakeModel("retry_once")
        evaluator = GeminiEvaluator(self.retriever, model)
        _, evaluation, attempts = retrieve_and_evaluate(self.retriever, evaluator, self.criteria, "麵食")
        self.assertEqual(evaluation.decision, "accept")
        self.assertEqual([a.query for a in attempts], ["麵食", "麵食 牛肉麵 鍋燒意麵 拉麵 義大利麵 羹麵"])
        self.assertTrue(all(a.criteria == self.criteria for a in attempts))
        self.assertEqual(len(model.calls), 2)

    def test_retry_expansion_changes_ranking_without_changing_eligibility(self):
        model = FakeModel("retry_once")
        _, _, attempts = retrieve_and_evaluate(self.retriever, GeminiEvaluator(self.retriever, model), self.criteria, "麵食")
        before, after = ([c.restaurant.id for c in a.retrieval.candidates] for a in attempts)
        self.assertNotEqual(before, after)
        self.assertEqual(set(before), set(after))
        # A sub-category shop (牛肉麵 / 鍋燒意麵) rises once sub-categories are searched.
        self.assertNotEqual(after[0], before[0])
        self.assertIn(after[0], {"DL-028", "DL-032"})

    def test_rewrites_are_narrower_and_never_add_excluded_cuisines(self):
        self.assertEqual(allowed_rewrites("麵食"), ("麵食 牛肉麵 鍋燒意麵 拉麵 義大利麵 羹麵",))
        no_ramen = SearchCriteria(cuisines=("麵食",), excluded_cuisines=("拉麵",))
        self.assertNotIn("拉麵", allowed_rewrites("麵食", no_ramen)[0])
        for query in ("火鍋", "牛肉麵", "斗六市", "麵類", "麵食 牛肉麵 鍋燒意麵 拉麵 義大利麵 羹麵"):
            with self.subTest(query=query):
                expected = () if query != "麵類" else ("麵類 牛肉麵 鍋燒意麵 拉麵 義大利麵 羹麵",)
                self.assertEqual(allowed_rewrites(query), expected)

    def test_repeated_retry_stops_at_upper_limit(self):
        model = FakeModel("retry")
        with patch("food_crag.evaluator.allowed_rewrites", lambda query, criteria=None: (query + " 牛肉麵",)):
            _, result, attempts = retrieve_and_evaluate(self.retriever, GeminiEvaluator(self.retriever, model), self.criteria, "麵食")
        self.assertEqual(result.decision, "stop")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(model.calls), 2)
        self.assertIn("上限", result.reasons[0])
        self.assertEqual(result.accepted_ids, ())

    def test_retry_after_expansion_is_rejected(self):
        model = FakeModel("retry")
        _, result, attempts = retrieve_and_evaluate(self.retriever, GeminiEvaluator(self.retriever, model), self.criteria, "麵食")
        self.assertEqual((result.decision, result.evidence_state, len(attempts), len(model.calls)), ("stop", "error", 2, 2))
        self.assertEqual(result.accepted_ids, ())

    def test_error_stops_without_retry_or_local_accept_fallback(self):
        model = FakeModel("error")
        _, result, attempts = retrieve_and_evaluate(self.retriever, GeminiEvaluator(self.retriever, model), self.criteria, "麵食")
        self.assertEqual(result.evidence_state, "error")
        self.assertEqual(result.accepted_ids, ())
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(model.calls), 1)


class AdapterTests(unittest.TestCase):
    def test_structured_rest_body_and_header_only_key(self):
        settings = Settings(gemini_api_key="test-secret")
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"ok":true}'}]}}]}).encode()
        with patch("food_crag.adapters.gemini.urlopen", return_value=response) as send:
            self.assertEqual(GeminiLanguageModel(settings).generate_json({"test": 1}, SCHEMA), {"ok": True})
        request = send.call_args.args[0]
        self.assertNotIn("test-secret", request.full_url)
        self.assertNotIn(b"test-secret", request.data)
        self.assertEqual(json.loads(request.data)["generationConfig"]["responseMimeType"], "application/json")
        self.assertNotIn("test-secret", repr(settings))

    def test_http_error_timeout_and_malformed_response_are_sanitized(self):
        model = GeminiLanguageModel(Settings(gemini_api_key="secret"))
        error = HTTPError("https://example.com", 429, "secret", {}, BytesIO(b"secret"))
        for failure, code in [(error, "http_429"), (TimeoutError("secret"), "network_or_timeout")]:
            with patch("food_crag.adapters.gemini.urlopen", side_effect=failure):
                with self.assertRaisesRegex(GeminiError, code) as caught:
                    model.generate_json({}, SCHEMA)
                self.assertNotIn("secret", str(caught.exception))
        for raw in (b"not JSON", b'{}', b'{"candidates": []}'):
            response = MagicMock()
            response.__enter__.return_value.read.return_value = raw
            with patch("food_crag.adapters.gemini.urlopen", return_value=response):
                with self.assertRaises(GeminiError):
                    model.generate_json({}, SCHEMA)

    def test_connection_drop_while_reading_response_stops_safely(self):
        failures = (http.client.RemoteDisconnected("secret"), ConnectionResetError("secret"),
                    http.client.IncompleteRead(b"secret"))
        model = GeminiLanguageModel(Settings(gemini_api_key="secret"))
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with patch("food_crag.adapters.gemini.urlopen", side_effect=failure):
                    with self.assertRaisesRegex(GeminiError, "network_or_timeout") as caught:
                        model.generate_json({}, SCHEMA)
                self.assertNotIn("secret", str(caught.exception))
                response = MagicMock()
                response.__enter__.return_value.read.side_effect = failure
                with patch("food_crag.adapters.gemini.urlopen", return_value=response):
                    with self.assertRaisesRegex(GeminiError, "network_or_timeout"):
                        model.generate_json({}, SCHEMA)
        pipeline = RecommendationPipeline(settings=Settings(gemini_api_key="secret"))
        with patch("food_crag.adapters.gemini.urlopen", side_effect=failures[0]):
            evaluation = pipeline.handle("u", "想吃麵，不要火鍋").evaluation
        self.assertEqual((evaluation.decision, evaluation.evidence_state, evaluation.model_called),
                         ("stop", "error", True))

    def test_missing_key_never_sends_request(self):
        with patch("food_crag.adapters.gemini.urlopen") as send:
            with self.assertRaisesRegex(GeminiError, "not_configured"):
                GeminiLanguageModel(Settings()).generate_json({}, SCHEMA)
            send.assert_not_called()


class ConfigurationTests(unittest.TestCase):
    def test_explicit_env_file_and_process_precedence_without_expansion(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text('GEMINI_API_KEY="local-secret"\nGEMINI_MODEL=\nAPP_PORT=8001\n', encoding="utf-8")
            settings = Settings.from_env(path)
            self.assertEqual(settings.gemini_api_key, "local-secret")
            self.assertEqual(settings.gemini_model, "gemini-3.5-flash-lite")
            self.assertEqual(settings.port, 8001)
            self.assertEqual(Settings.from_env().gemini_api_key, "")
            with patch.dict(os.environ, {"GEMINI_API_KEY": "process-secret"}):
                self.assertEqual(Settings.from_env(path).gemini_api_key, "process-secret")


class EvaluationHTTPTests(unittest.TestCase):
    def test_real_http_mocked_model_pass_and_insufficient_data_have_traces(self):
        model = FakeModel()
        pipeline = RecommendationPipeline(model=model)
        server = create_server(port=0, pipeline=pipeline)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for message, expected, ids in [("想吃麵", "accept", ["DL-033"]), ("50元以內", "accept", ["DL-033"])]:
                request = Request(f"http://127.0.0.1:{server.server_port}/api/chat", data=json.dumps({"user_id": "A", "message": message}).encode(), headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=5) as response:
                    data = json.load(response)
                self.assertEqual(data["evaluation"]["decision"], expected)
                self.assertEqual(len(data["attempts"]), 1)
                self.assertEqual(data["restaurant_ids"], ids)
            self.assertEqual(len(model.calls), 2)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main()
