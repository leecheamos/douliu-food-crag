"""Embedding re-ranking with fake vectors; the paid embedding API is never called."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from food_crag.adapters.gemini import GeminiEmbedder, GeminiError
from food_crag.config import Settings
from food_crag.dataset import load_restaurants
from food_crag.domain import SearchCriteria
from food_crag.embedding import (
    EmbeddingCacheError, RRF_K, SemanticIndex, build_cache, load_from_settings, load_index, rrf_scores,
)
from food_crag.evaluator import GeminiEvaluator
from food_crag.pipeline import RecommendationPipeline
from food_crag.retriever import RestaurantRetriever, document_text

from test_evaluator import FakeModel
from test_retriever import fixture


class FakeEmbedder:
    """Query vector is fixed; document vectors come from a per-text table."""

    def __init__(self, documents=None, query=(1.0, 0.0), fail=False):
        self.documents = documents or {}
        self.query = list(query)
        self.fail = fail
        self.calls = []

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        self.calls.append((task_type, list(texts)))
        if self.fail:
            raise GeminiError("network_or_timeout")
        if task_type == "RETRIEVAL_QUERY":
            return [list(self.query) for _ in texts]
        return [list(self.documents.get(text, (0.0, 1.0))) for text in texts]


def noodle_shops():
    # Same cuisine, so every shop passes hard filters and has a lexical hit.
    return [fixture("A", name="甲麵店"), fixture("B", name="乙麵店"), fixture("C", name="丙麵店")]


def semantic_index(restaurants, vectors, embedder=None):
    return SemanticIndex("gemini-test", {r.id: list(vectors[r.id]) for r in restaurants},
                         embedder or FakeEmbedder())


class FusionTests(unittest.TestCase):
    def test_rrf_rewards_agreement_between_rankings(self):
        scores = rrf_scores([["A", "B", "C"], ["C", "A", "B"]])
        self.assertAlmostEqual(scores["A"], 1 / (RRF_K + 1) + 1 / (RRF_K + 2))
        self.assertGreater(scores["A"], scores["C"])
        self.assertGreater(scores["C"], scores["B"])

    def test_hybrid_reranks_but_never_changes_the_candidate_set(self):
        restaurants = noodle_shops()
        lexical = RestaurantRetriever(restaurants).search("麵食", SearchCriteria())
        # Make the lexically last shop the semantic best by a wide margin.
        last = lexical.candidates[-1].restaurant.id
        vectors = {r.id: ((1.0, 0.0) if r.id == last else (0.0, 1.0)) for r in restaurants}
        hybrid = RestaurantRetriever(restaurants, semantic_index(restaurants, vectors)).search("麵食", SearchCriteria())
        self.assertEqual(lexical.ranking, "lexical")
        self.assertEqual(hybrid.ranking, "hybrid_rrf")
        self.assertEqual({c.restaurant.id for c in hybrid.candidates}, {c.restaurant.id for c in lexical.candidates})
        self.assertEqual(hybrid.decisions, lexical.decisions)
        self.assertNotEqual(hybrid.candidates[-1].restaurant.id, last)
        for candidate in hybrid.candidates:
            self.assertIsNotNone(candidate.semantic_score)
            self.assertGreater(candidate.lexical_score, 0)
            self.assertTrue(candidate.matched_terms)

    def test_hard_filters_still_apply_before_semantic_ranking(self):
        restaurants = [*noodle_shops(), fixture("HOT", name="火鍋店", cuisines=("火鍋",))]
        vectors = {r.id: ((1.0, 0.0) if r.id == "HOT" else (0.0, 1.0)) for r in restaurants}
        retriever = RestaurantRetriever(restaurants, semantic_index(restaurants, vectors))
        result = retriever.search("麵食", SearchCriteria(excluded_cuisines=("火鍋",)))
        self.assertNotIn("HOT", {c.restaurant.id for c in result.candidates})

    def test_query_embedding_failure_falls_back_to_lexical_order(self):
        restaurants = noodle_shops()
        vectors = {r.id: (1.0, 0.0) for r in restaurants}
        failing = RestaurantRetriever(restaurants, semantic_index(restaurants, vectors, FakeEmbedder(fail=True)))
        lexical = RestaurantRetriever(restaurants).search("麵食", SearchCriteria())
        result = failing.search("麵食", SearchCriteria())
        self.assertEqual(result.ranking, "lexical_fallback")
        self.assertEqual(result.candidates, lexical.candidates)

    def test_query_embeddings_are_cached(self):
        restaurants = noodle_shops()
        embedder = FakeEmbedder()
        retriever = RestaurantRetriever(restaurants, semantic_index(restaurants, {r.id: (1.0, 0.0) for r in restaurants}, embedder))
        for _ in range(3):
            retriever.search("麵食", SearchCriteria())
        self.assertEqual(len(embedder.calls), 1)

    def test_index_must_cover_exactly_the_loaded_restaurants(self):
        restaurants = noodle_shops()
        partial = SemanticIndex("gemini-test", {"A": [1.0, 0.0]}, FakeEmbedder())
        with self.assertRaises(ValueError):
            RestaurantRetriever(restaurants, partial)


class HybridEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.restaurants = load_restaurants()
        vectors = {r.id: (float(i % 3), 1.0) for i, r in enumerate(self.restaurants)}
        self.retriever = RestaurantRetriever(self.restaurants, semantic_index(self.restaurants, vectors))
        self.criteria = SearchCriteria(cuisines=("麵食",), excluded_cuisines=("火鍋",))
        self.report = self.retriever.search("麵食", self.criteria)
        self.model = FakeModel()
        self.evaluator = GeminiEvaluator(self.retriever, self.model)

    def test_hybrid_report_is_verified_and_sent_with_ranking_note(self):
        self.assertEqual(self.report.ranking, "hybrid_rrf")
        evaluation = self.evaluator.evaluate(self.criteria, self.report)
        self.assertEqual(evaluation.decision, "accept")
        payload = self.model.calls[0]
        self.assertEqual(payload["ranking"], "hybrid_rrf")
        self.assertIn("not evidence", payload["limits"])
        self.assertIn("semantic_score", payload["candidates"][0])

    def test_forged_semantic_scores_or_order_are_vetoed(self):
        first = self.report.candidates[0]
        forged = [
            (replace(first, semantic_score=0.999), *self.report.candidates[1:]),
            (replace(first, semantic_score=None), *self.report.candidates[1:]),
            (replace(first, score=first.score * 2), *self.report.candidates[1:]),
            (replace(first, lexical_score=first.lexical_score / 2), *self.report.candidates[1:]),
            tuple(reversed(self.report.candidates)),
        ]
        for candidates in forged:
            with self.subTest(candidates=[c.restaurant.id for c in candidates]):
                result = self.evaluator.evaluate(self.criteria, replace(self.report, candidates=candidates))
                self.assertEqual((result.decision, result.evidence_state), ("stop", "error"))
        self.assertEqual(self.model.calls, [])

    def test_ranking_mode_must_match_local_configuration(self):
        lexical_only = GeminiEvaluator(RestaurantRetriever(self.restaurants), FakeModel())
        result = lexical_only.evaluate(self.criteria, self.report)
        self.assertEqual((result.decision, result.evidence_state), ("stop", "error"))
        claimed_lexical = replace(self.report, ranking="lexical")
        result = self.evaluator.evaluate(self.criteria, claimed_lexical)
        self.assertEqual((result.decision, result.evidence_state), ("stop", "error"))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.restaurants = noodle_shops()
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "gemini-test.json"
        self.embedder = FakeEmbedder({document_text(r): (1.0, float(i)) for i, r in enumerate(self.restaurants)})

    def tearDown(self):
        self.directory.cleanup()

    def test_round_trip_uses_document_task_type(self):
        build_cache(self.restaurants, self.embedder, "gemini-test", self.path)
        self.assertEqual(self.embedder.calls[0][0], "RETRIEVAL_DOCUMENT")
        index = load_index(self.restaurants, self.embedder, "gemini-test", self.path)
        self.assertEqual(index.vectors["B"], [1.0, 1.0])

    def test_missing_stale_mismatched_or_invalid_cache_is_rejected(self):
        with self.assertRaisesRegex(EmbeddingCacheError, "cache_missing"):
            load_index(self.restaurants, self.embedder, "gemini-test", self.path)
        build_cache(self.restaurants, self.embedder, "gemini-test", self.path)
        changed = [replace(self.restaurants[0], menu_summary="新菜單"), *self.restaurants[1:]]
        with self.assertRaisesRegex(EmbeddingCacheError, "cache_stale"):
            load_index(changed, self.embedder, "gemini-test", self.path)
        with self.assertRaisesRegex(EmbeddingCacheError, "cache_model_mismatch"):
            load_index(self.restaurants, self.embedder, "gemini-other", self.path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["documents"]["A"]["vector"] = ["x"]
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(EmbeddingCacheError, "cache_invalid"):
            load_index(self.restaurants, self.embedder, "gemini-test", self.path)

    def test_missing_cache_disables_embeddings_without_breaking_chat(self):
        settings = Settings(gemini_api_key="secret", embedding_model="gemini-not-built")
        index, status = load_from_settings(self.restaurants, settings)
        self.assertIsNone(index)
        self.assertEqual(status, "cache_missing")
        with patch("food_crag.adapters.gemini.urlopen") as send:
            pipeline = RecommendationPipeline(settings=settings, model=FakeModel())
            result = pipeline.handle("u", "想吃麵，不要火鍋")
        send.assert_not_called()
        self.assertEqual(pipeline.embedding_status, "cache_missing")
        self.assertEqual(result.retrieval.ranking, "lexical")
        self.assertEqual(result.evaluation.decision, "accept")


class EmbedderAdapterTests(unittest.TestCase):
    def response(self, payload):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        return response

    def test_batch_request_body_task_type_and_header_only_key(self):
        settings = Settings(gemini_api_key="test-secret", embedding_model="gemini-embedding-001")
        embedder = GeminiEmbedder(settings, dimensions=2)
        reply = self.response({"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]})
        with patch("food_crag.adapters.gemini.urlopen", return_value=reply) as send:
            vectors = embedder.embed(["甲", "乙"], task_type="RETRIEVAL_QUERY")
        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        request = send.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/models/gemini-embedding-001:batchEmbedContents"))
        self.assertNotIn("test-secret", request.full_url)
        self.assertNotIn(b"test-secret", request.data)
        body = json.loads(request.data)
        self.assertEqual([r["taskType"] for r in body["requests"]], ["RETRIEVAL_QUERY"] * 2)
        self.assertEqual(body["requests"][0]["model"], "models/gemini-embedding-001")
        self.assertEqual(body["requests"][0]["outputDimensionality"], 2)

    def test_large_inputs_are_split_into_batches(self):
        settings = Settings(gemini_api_key="k", embedding_model="gemini-embedding-001")
        embedder = GeminiEmbedder(settings, dimensions=1)
        replies = [self.response({"embeddings": [{"values": [1.0]}] * 40}),
                   self.response({"embeddings": [{"values": [1.0]}] * 40}),
                   self.response({"embeddings": [{"values": [2.0]}] * 25})]
        with patch("food_crag.adapters.gemini.urlopen", side_effect=replies) as send,                 patch("food_crag.adapters.gemini.time.sleep") as pause:
            vectors = embedder.embed(["x"] * 105)
        self.assertEqual(send.call_count, 3)
        self.assertEqual(pause.call_count, 2)  # Free-tier pause between batches only.
        self.assertEqual(len(vectors), 105)

    def test_unconfigured_invalid_or_malformed_fail_closed(self):
        with patch("food_crag.adapters.gemini.urlopen") as send:
            with self.assertRaisesRegex(GeminiError, "not_configured"):
                GeminiEmbedder(Settings(gemini_api_key="k")).embed(["x"])
            with self.assertRaisesRegex(GeminiError, "invalid_model_name"):
                GeminiEmbedder(Settings(gemini_api_key="k", embedding_model="../bad")).embed(["x"])
            send.assert_not_called()
        embedder = GeminiEmbedder(Settings(gemini_api_key="secret", embedding_model="gemini-embedding-001"), dimensions=2)
        for payload in ({}, {"embeddings": []}, {"embeddings": [{"values": [1.0]}]}):
            with patch("food_crag.adapters.gemini.urlopen", return_value=self.response(payload)):
                with self.assertRaises(GeminiError) as caught:
                    embedder.embed(["x"])
                self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
