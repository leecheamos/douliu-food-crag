"""Retrieval behavior: synthetic price/time fixtures are not restaurant facts."""
import json
from math import log
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from food_crag.dataset import load_restaurants
from food_crag.domain import Restaurant, SearchCriteria
from food_crag.pipeline import RecommendationPipeline
from food_crag.retriever import RestaurantRetriever, cuisine_labels, fitting_items, item_labels
from food_crag.server import create_server
from food_crag.tfidf import calculate_idf, calculate_tf, cosine_similarity, tokenize


def fixture(identifier="TEST-1", **changes):
    """Artificial records used only for conditions absent from the real data."""
    values = dict(id=identifier, name="測試麵店", county="雲林縣", district="斗六市",
                  address="測試用地址", source_urls=(), checked_at=None,
                  cuisines=("麵食",), price_min_twd=100, price_max_twd=150,
                  price_evidence="測試用主餐標價區間，非真實店家資料",
                  meal_periods=("lunch",), landmarks=("鎮南國小站",),
                  sources=({"supports": ["price_evidence", "meal_periods", "landmarks"]},))
    values.update(changes)
    return Restaurant(**values)


class ItemPriceTests(unittest.TestCase):
    """Budget is checked on the requested kind of dish when item prices exist (2026-11-23)."""

    def test_requested_kind_must_fit_budget_not_any_cheap_item(self):
        rice_cheap = fixture(menu_items=("滷肉飯", "牛肉麵"), menu_prices=(("滷肉飯", 40), ("牛肉麵", 120)),
                             price_min_twd=40, price_max_twd=120)
        noodles = SearchCriteria(cuisines=("麵食",), budget_twd=100)
        decision = RestaurantRetriever.filter(rice_cheap, noodles)
        self.assertEqual((decision.outcome, decision.reasons), ("excluded", ("符合的品項標價都超出預算",)))
        self.assertEqual(RestaurantRetriever.filter(rice_cheap, SearchCriteria(cuisines=("麵食",), budget_twd=120)).outcome,
                         "eligible")
        # No item rule for the type (火鍋) or no item prices: the shop's cheapest main dish decides.
        self.assertIsNone(fitting_items(rice_cheap, SearchCriteria(cuisines=("火鍋",))))
        self.assertEqual(RestaurantRetriever.filter(fixture(price_min_twd=40), noodles).outcome, "eligible")

    def test_item_kinds(self):
        shop = fixture(menu_items=("滷肉飯", "陽春麵", "米粉湯", "牛肉乾拌麵", "蛋餅"),
                       menu_prices=(("滷肉飯", 30), ("陽春麵", 40), ("米粉湯", 45), ("牛肉乾拌麵", 90), ("蛋餅", 35)))
        names = lambda **c: [i for i, _ in fitting_items(shop, SearchCriteria(**c))]
        self.assertEqual(names(cuisines=("麵食",)), ["陽春麵", "米粉湯", "牛肉乾拌麵"])
        self.assertEqual(names(cuisines=("飯食",)), ["滷肉飯"])
        self.assertEqual(names(cuisines=("牛肉麵",)), ["牛肉乾拌麵"])
        self.assertEqual(names(dishes=("滷肉飯",)), ["滷肉飯"])
        self.assertEqual(len(names()), 5)

    def test_real_data_noodles_under_50_drop_the_brunch_shop(self):
        retriever = RestaurantRetriever(load_restaurants())
        result = retriever.search("麵食", SearchCriteria(cuisines=("麵食",), budget_twd=50))
        ids = {c.restaurant.id for c in result.candidates}
        self.assertNotIn("DL-031", ids)  # 蛋餅 35 元, but its noodles start at 90 元.
        self.assertIn("DL-028", ids)


class ItemLabelTests(unittest.TestCase):
    """Menu items join specific kinds (user decisions 2026-11-23); 麵食/飯食 stay shop-level."""
    @classmethod
    def setUpClass(cls):
        cls.restaurants = load_restaurants()
        cls.catalog = {r.id: r for r in cls.restaurants}
        cls.retriever = RestaurantRetriever(cls.restaurants)

    def found(self, cuisine, budget=None):
        report = self.retriever.search(cuisine, SearchCriteria(cuisines=(cuisine,), budget_twd=budget), limit=40)
        return {c.restaurant.id for c in report.candidates}

    def test_keywords_and_hand_labels(self):
        labels = lambda shop, item: item_labels(self.catalog[shop], item)
        self.assertEqual(labels("DL-033", "鍋燒泡飯"), {"飯食"})  # 鍋燒 is noodles, except with rice.
        self.assertIn("麵食", labels("DL-032", "韓式部隊鍋燒"))
        self.assertNotIn("火鍋", labels("DL-032", "韓式部隊鍋燒"))
        self.assertEqual(labels("DL-019", "唐揚雞"), {"炸物"})
        self.assertTrue({"羹類", "麵食"} <= labels("DL-009", "魷魚羹麵"))
        self.assertEqual(labels("DL-030", "一品牛五花"), {"火鍋"})
        self.assertEqual(labels("DL-028", "牛肉湯麵"), {"麵食"})  # Hand label: no beef in it.
        self.assertEqual(labels("DL-031", "奶油麵包"), {"甜點"})  # Not in any menu yet; rule only.
        self.assertIn("羹類", cuisine_labels(("羹麵",)))

    def test_shops_found_through_their_menu(self):
        self.assertEqual(self.found("炒飯"), {"DL-013", "DL-018"})
        self.assertIn("DL-019", self.found("炸物"))
        self.assertTrue({"DL-015", "DL-026"} <= self.found("火鍋"))
        self.assertTrue({"DL-012", "DL-015", "DL-031"} <= self.found("義大利麵"))
        self.assertEqual(self.found("羹類"), {"DL-002", "DL-004", "DL-008", "DL-009"})
        self.assertNotIn("DL-006", self.found("麵食"))  # 羊肉麵線 alone does not make a noodle shop.

    def test_hand_labelled_item_needs_the_literal_dish(self):
        self.assertFalse(self.found("牛肉麵", 50))  # 牛肉湯麵 50 元 has no beef: the real ones cost more.
        report = self.retriever.search("牛肉湯麵", SearchCriteria(dishes=("牛肉湯麵",)))
        self.assertIn("DL-028", {c.restaurant.id for c in report.candidates})


class MathTests(unittest.TestCase):
    def test_ported_formulas_and_empty_inputs(self):
        self.assertEqual(calculate_tf([]), {})
        self.assertEqual(calculate_tf(["a", "a", "b"]), {"a": 2 / 3, "b": 1 / 3})
        idf = calculate_idf([["a", "a"], ["a", "b"]])
        self.assertEqual(idf["a"], 1)
        self.assertAlmostEqual(idf["b"], log(3 / 2) + 1)
        self.assertEqual(calculate_idf([]), {})
        self.assertEqual(cosine_similarity({}, {"a": 1}), 0)
        self.assertAlmostEqual(cosine_similarity({"a": 1}, {"a": 3, "b": 4}), .6)
        self.assertEqual(cosine_similarity({"a": 1}, {"b": 1}), 0)

    def test_food_tokenization_removes_academic_normalization(self):
        self.assertEqual(tokenize("麵店"), tokenize("麵食"))
        self.assertIn("牛肉", tokenize("牛肉麵"))
        self.assertIn("博士", tokenize("博士生"))
        self.assertNotIn("學生", tokenize("博士生"))


class RetrievalTests(unittest.TestCase):
    def test_all_hard_conditions_apply_before_ranking(self):
        good = fixture("OK")
        restaurants = [good,
            fixture("AREA", district="虎尾鎮"),
            fixture("CUISINE", cuisines=("火鍋",), name="麵麵麵火鍋"),
            fixture("EXCLUDE", cuisines=("麵食", "火鍋")),
            fixture("PRICE", price_min_twd=300, price_max_twd=400),
            fixture("TIME", meal_periods=("dinner",)),
            fixture("PLACE", landmarks=("斗六車站",)),
        ]
        criteria = SearchCriteria(cuisines=("麵食",), excluded_cuisines=("火鍋",),
                                  budget_twd=200, meal="lunch", landmark="鎮南國小站")
        result = RestaurantRetriever(restaurants).search("麵食", criteria)
        self.assertEqual([c.restaurant.id for c in result.candidates], ["OK"])
        self.assertTrue(all(d.outcome == "excluded" for d in result.decisions if d.restaurant_id != "OK"))

    def test_unknown_hard_fields_never_pass(self):
        changes = [dict(price_min_twd=None, price_max_twd=None),
                   dict(price_min_twd=None), dict(price_evidence=None), dict(sources=()),
                   dict(landmarks=()), dict(cuisines=())]
        criteria = SearchCriteria(cuisines=("麵食",), budget_twd=200, meal="lunch", landmark="鎮南國小站")
        for change in changes:
            with self.subTest(change=change):
                result = RestaurantRetriever([fixture(**change)]).search("麵食", criteria)
                self.assertEqual(result.status, "insufficient_data")
                self.assertEqual(result.candidates, ())
                self.assertEqual(result.decisions[0].outcome, "unknown")
                self.assertTrue(result.decisions[0].reasons)

    def test_meal_is_a_preference_but_sourced_mismatch_excludes(self):
        criteria = SearchCriteria(cuisines=("麵食",), meal="dinner")
        unknown = RestaurantRetriever([fixture(meal_periods=None)]).search("麵食", criteria)
        self.assertEqual(len(unknown.candidates), 1)
        mismatch = RestaurantRetriever([fixture(meal_periods=("lunch",))]).search("麵食", criteria)
        self.assertEqual(mismatch.decisions[0].outcome, "excluded")

    def test_budget_matches_when_cheapest_main_dish_fits(self):
        # Prices are listed main-dish single-item prices, not per-person meal cost.
        retriever = RestaurantRetriever([fixture(price_min_twd=100, price_max_twd=250)])
        self.assertEqual(len(retriever.search("麵食", SearchCriteria(budget_twd=100)).candidates), 1)
        result = retriever.search("麵食", SearchCriteria(budget_twd=99))
        self.assertEqual(result.status, "no_results")
        self.assertIn("最便宜的主餐標價超出預算", result.decisions[0].reasons)

    def test_specific_cuisine_and_exclusions_do_not_guess_broad_labels(self):
        retriever = RestaurantRetriever([fixture()])
        result = retriever.search("牛肉麵", SearchCriteria(cuisines=("牛肉麵",)))
        self.assertEqual(result.status, "insufficient_data")
        result = retriever.search("麵食", SearchCriteria(excluded_cuisines=("牛肉麵",)))
        self.assertEqual(result.status, "insufficient_data")
        result = RestaurantRetriever([fixture(cuisines=("牛肉麵",))]).search("麵食", SearchCriteria(excluded_cuisines=("麵食",)))
        self.assertEqual(result.status, "no_results")

    def test_evidence_and_score_agree_and_ties_are_stable(self):
        retriever = RestaurantRetriever([fixture("B"), fixture("A")])
        result = retriever.search("麵食", SearchCriteria(), limit=1)
        self.assertEqual(result.candidates[0].restaurant.id, "A")
        candidate = result.candidates[0]
        self.assertGreater(candidate.score, 0)
        self.assertLessEqual(candidate.score, 1 + 1e-12)
        self.assertIn("麵食", candidate.matched_terms)
        self.assertEqual(candidate.query_coverage, 1)
        self.assertTrue(candidate.evidence)
        self.assertEqual(candidate.restaurant.sources, fixture().sources)
        self.assertEqual(result.eligible_count, 2)
        self.assertEqual(len(result.decisions), 2)

    def test_lexical_score_orders_eligible_candidates(self):
        retriever = RestaurantRetriever([
            fixture("LOW", name="測試海鮮熱炒簡餐店", cuisines=("麵食", "海鮮", "熱炒", "簡餐")),
            fixture("HIGH", name="麵食 麵食 麵食"),
        ])
        result = retriever.search("麵食", SearchCriteria(cuisines=("麵食",)))
        self.assertEqual([c.restaurant.id for c in result.candidates], ["HIGH", "LOW"])
        self.assertGreater(result.candidates[0].score, result.candidates[1].score)

    def test_empty_unknown_query_and_zero_scores(self):
        empty = RestaurantRetriever([]).search("麵食", SearchCriteria())
        self.assertEqual(empty.status, "empty_dataset")
        for query in ("", " ", "zzzzunseentoken"):
            with self.subTest(query=query):
                result = RestaurantRetriever([fixture()]).search(query, SearchCriteria())
                self.assertEqual(result.status, "no_results")
                self.assertEqual(result.candidates, ())
                self.assertEqual(result.decisions[0].outcome, "no_lexical_match")

    def test_invalid_limit_budget_and_duplicate_ids_fail_explicitly(self):
        retriever = RestaurantRetriever([fixture()])
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                retriever.search("麵食", SearchCriteria(), limit=limit)
        for budget in (0, -1, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                retriever.search("麵食", SearchCriteria(budget_twd=budget))
        with self.assertRaises(ValueError):
            RestaurantRetriever([fixture(), fixture()])

    def test_index_is_reused_across_queries(self):
        retriever = RestaurantRetriever([fixture()])
        with patch("food_crag.retriever.calculate_idf", side_effect=AssertionError("index rebuilt")):
            retriever.search("麵食", SearchCriteria())
            retriever.search("斗六", SearchCriteria())

    def test_real_dataset_price_and_service_gaps_remain_visible(self):
        retriever = RestaurantRetriever(load_restaurants())
        noodles = retriever.search("麵食", SearchCriteria(cuisines=("麵食",)))
        self.assertEqual(noodles.status, "candidates_found")
        self.assertTrue(all("麵食" in cuisine_labels(c.restaurant.cuisines) for c in noodles.candidates))
        self.assertIn("DL-017", {c.restaurant.id for c in noodles.candidates})
        budget = retriever.search("麵食", SearchCriteria(cuisines=("麵食",), budget_twd=50))
        self.assertEqual(budget.status, "candidates_found")
        self.assertTrue(all(c.restaurant.price_min_twd <= 50 for c in budget.candidates))
        self.assertLess(len(budget.candidates), len(noodles.candidates))
        self.assertFalse(retriever.search("麵食", SearchCriteria(cuisines=("麵食",), budget_twd=20)).candidates)
        # Meal is a preference: unknown service periods do not block (reply reminds to check hours).
        time = retriever.search("斗六", SearchCriteria(meal="lunch"))
        self.assertEqual(time.status, "candidates_found")
        self.assertFalse(any(d.outcome == "unknown" for d in time.decisions))
        place = retriever.search("鎮南國小站", SearchCriteria(landmark="鎮南國小站"))
        self.assertEqual([c.restaurant.id for c in place.candidates], ["DL-025"])


class RetrievalIntegrationTests(unittest.TestCase):
    def test_accumulated_positive_query_does_not_rank_negative_or_update_words(self):
        pipeline = RecommendationPipeline()
        pipeline.handle("A", "想吃麵，不要火鍋")
        result = pipeline.handle("A", "50元以內")
        self.assertEqual(result.retrieval.query, "麵食")
        self.assertTrue(all(c.restaurant.price_min_twd <= 50 for c in result.retrieval.candidates))
        result = pipeline.handle("A", "取消預算限制")
        self.assertEqual(result.retrieval.status, "candidates_found")
        self.assertEqual(result.restaurant_ids, ())
        self.assertTrue(all("火鍋" not in cuisine_labels(c.restaurant.cuisines) for c in result.retrieval.candidates))
        self.assertIsNone(pipeline.handle("A", "想吃火鍋").retrieval)
        self.assertIsNone(pipeline.handle("A", "重設").retrieval)

    def test_no_criteria_relaxation_or_retry(self):
        retriever = RestaurantRetriever([fixture(price_min_twd=300, price_max_twd=400)])
        pipeline = RecommendationPipeline(retriever)
        with patch.object(retriever, "search", wraps=retriever.search) as search:
            result = pipeline.handle("A", "預算200元，想吃麵")
            self.assertEqual(search.call_count, 1)
            self.assertEqual(result.criteria.budget_twd, 200)
            self.assertEqual(result.retrieval.status, "no_results")
            self.assertFalse(result.retrieval.candidates)

    def test_real_http_candidates_budget_gap_and_user_isolation(self):
        server = create_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def send(user, message):
            request = Request(f"http://127.0.0.1:{server.server_port}/api/chat",
                              data=json.dumps({"user_id": user, "message": message}).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                return json.load(response)

        try:
            first = send("A", "想吃麵，不要火鍋")
            self.assertEqual(first["retrieval"]["status"], "candidates_found")
            candidate = first["retrieval"]["candidates"][0]
            self.assertTrue(candidate["restaurant"]["sources"])
            self.assertTrue(candidate["matched_terms"])
            self.assertEqual(first["restaurant_ids"], [])
            budget = send("A", "50元以內")
            self.assertEqual(budget["criteria"]["cuisines"], ["麵食"])
            self.assertTrue(all(c["restaurant"]["price_min_twd"] <= 50 for c in budget["retrieval"]["candidates"]))
            self.assertTrue(any(d["outcome"] == "excluded" for d in budget["retrieval"]["decisions"]))
            other = send("B", "想吃火鍋")
            self.assertEqual(other["retrieval"]["status"], "candidates_found")
            self.assertIsNone(other["criteria"]["budget_twd"])
            self.assertIsNone(send("A", "重設")["retrieval"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
