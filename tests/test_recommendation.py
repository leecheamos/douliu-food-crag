"""Step 7: grounded template replies, verification and readable fallbacks."""
from dataclasses import replace
import unittest

from food_crag.dataset import load_restaurants
from food_crag.domain import Evaluation, SearchCriteria
from food_crag.pipeline import RecommendationPipeline
from food_crag.recommendation import (MAX_RECOMMENDATIONS, Recommendation, TemplateReplyGenerator,
                                      verify_recommendation)
from food_crag.retriever import RestaurantRetriever


class AcceptModel:
    """Accepts every candidate it is shown, to exercise the 3-restaurant cap."""
    def __init__(self, error=None):
        self.error = error

    def generate_json(self, payload, schema):
        if self.error:
            raise self.error
        return dict(decision="accept", evidence_state="sufficient", rewrite_query="", reasons=["測試"],
                    accepted_ids=[c["restaurant"]["id"] for c in payload["candidates"]])


class TemplateReplyTests(unittest.TestCase):
    def setUp(self):
        self.restaurants = load_restaurants()
        self.retriever = RestaurantRetriever(self.restaurants)
        self.catalog = {r.id: r for r in self.restaurants}
        self.criteria = SearchCriteria(cuisines=("麵食",), excluded_cuisines=("火鍋",))
        self.report = self.retriever.search("麵食", self.criteria)
        self.generator = TemplateReplyGenerator(self.restaurants)

    def accept(self, ids):
        return Evaluation("accept", ("測試",), "sufficient", tuple(ids), "", True)

    def test_accept_lists_grounded_fields_in_ranking_order(self):
        evaluation = self.accept(["DL-027", "DL-017"])
        result = self.generator.generate(self.criteria, self.report, evaluation)
        # menu_items joined the ranking text (2026-11-21); DL-027 lists more noodle dishes.
        self.assertEqual(result.restaurant_ids, ("DL-027", "DL-017"))
        for restaurant_id in result.restaurant_ids:
            r = self.catalog[restaurant_id]
            for fact in (r.name, f"地址：{r.address}", r.map_search_url):
                self.assertIn(fact, result.reply)
        # Conversational lead restating the conditions and each shop's signature dishes.
        self.assertTrue(result.reply.startswith("想吃麵食、不要火鍋的話，我推薦這 2 家："))
        self.assertIn("\n2.烹小鮮麵館，主要賣：豬油拌飯、乾拌麵和燒餅。", result.reply)
        self.assertIn("價格：主餐約 40～110 元", result.reply)
        self.assertIn("不是整餐費用", result.reply)
        self.assertNotIn("資料來源", result.reply)
        self.assertTrue(verify_recommendation(result, self.catalog, evaluation, self.report))

    def test_meal_preference_adds_opening_hours_reminder(self):
        dinner = replace(self.criteria, meal="dinner")
        report = self.retriever.search("麵食", dinner)
        result = self.generator.generate(dinner, report, self.accept(["DL-017"]))
        self.assertTrue(result.reply.startswith("想吃麵食、不要火鍋、吃晚餐的話"))
        self.assertIn("是否有供應晚餐尚未確認", result.reply)
        self.assertNotIn("供應晚餐", self.generator.generate(self.criteria, self.report, self.accept(["DL-017"])).reply)

    def test_at_most_three_and_never_padded(self):
        all_ids = [c.restaurant.id for c in self.report.candidates]
        self.assertGreater(len(all_ids), MAX_RECOMMENDATIONS)
        many = self.generator.generate(self.criteria, self.report, self.accept(all_ids))
        self.assertEqual(many.restaurant_ids, tuple(all_ids[:MAX_RECOMMENDATIONS]))
        one = self.generator.generate(self.criteria, self.report, self.accept(all_ids[:1]))
        self.assertEqual(len(one.restaurant_ids), 1)
        self.assertIn("我推薦這家：", one.reply)

    def test_known_price_is_shown_with_its_evidence(self):
        r = self.catalog["DL-017"]
        priced = replace(r, price_min_twd=90, price_max_twd=160, price_evidence="測試菜單標價",
                         sources=(*r.sources, dict(r.sources[0], supports=["price_evidence"])))
        retriever = RestaurantRetriever([priced])
        report = retriever.search("麵食", self.criteria)
        result = TemplateReplyGenerator([priced]).generate(self.criteria, report, self.accept(["DL-017"]))
        self.assertIn("價格：主餐約 90～160 元", result.reply)
        unpriced = replace(r, price_min_twd=None, price_max_twd=None, price_evidence=None, menu_prices=())
        result = TemplateReplyGenerator([unpriced]).generate(self.criteria, report, self.accept(["DL-017"]))
        self.assertIn("價格：未知", result.reply)

    def test_budget_lines_list_only_in_budget_items_of_the_requested_kind(self):
        criteria = replace(self.criteria, budget_twd=100)
        report = self.retriever.search("麵食", criteria)
        result = self.generator.generate(criteria, report, self.accept(["DL-027"]))
        # 紅燒牛肉麵 135 元 is the signature but over budget; 香妃鴨飯 is rice. Max 3, menu order.
        self.assertIn("\n1.賣麥面麵食館，100 元以內有：餛飩乾拌麵 85 元、餛飩湯麵 85 元、麻醬乾拌麵 60 元。", result.reply)
        self.assertNotIn("紅燒牛肉麵", result.reply.split("\n\n")[0])

    def test_no_shop_replies_are_conversational_with_next_steps(self):
        fried = [replace(r, menu_items=(), menu_prices=(), item_tags=()) for r in self.restaurants]
        fried_rice = SearchCriteria(cuisines=("炒飯",))  # Without menus no shop is 炒飯, rice shops are unknown.
        result = self.generator.generate(fried_rice, RestaurantRetriever(fried).search("炒飯", fried_rice),
                                         Evaluation("clarify", ("x",), "insufficient"))
        self.assertEqual(result.reply, "資料裡還沒有店被分到炒飯，我先不亂推薦。要不要換成飯食，或看看其他類型？")
        self.assertEqual(result.quick_replies, ("想吃飯食", "還有其他的嗎？"))
        steak = SearchCriteria(cuisines=("牛排",), budget_twd=300)  # 西堤 and 陶板屋's 牛排 items cost more.
        result = self.generator.generate(steak, self.retriever.search("牛排", steak),
                                         Evaluation("stop", ("x",), "no_results"))
        self.assertEqual(result.reply, "牛排的店有 2 家，但價格都超過 300 元，所以這次先不推薦，"
                                       "也沒有自己放寬你的條件。要不要取消預算限制，或換個類型？")
        self.assertEqual(result.quick_replies, ("取消預算限制", "還有其他的嗎？"))

    def test_insufficient_no_results_and_error_replies_are_readable_without_ids(self):
        budget = replace(self.criteria, budget_twd=200)
        unpriced = [replace(r, price_min_twd=None, price_max_twd=None, price_evidence=None, menu_prices=()) for r in self.restaurants]
        cases = [
            (budget, RestaurantRetriever(unpriced).search("麵食", budget), Evaluation("clarify", ("x",), "insufficient"),
             "這些店沒有確切的標價，沒辦法確定 200 元吃得到"),
            (self.criteria, self.report, Evaluation("stop", ("x",), "no_results"), "也沒有自己放寬你的條件"),
            (self.criteria, self.report, Evaluation("stop", ("Gemini 評估未完成：http_429",), "error", model_called=True),
             "請稍後再試"),
        ]
        for criteria, report, evaluation, expected in cases:
            with self.subTest(expected=expected):
                result = self.generator.generate(criteria, report, evaluation)
                self.assertEqual(result.restaurant_ids, ())
                self.assertIn(expected, result.reply)
                self.assertNotIn("http_429", result.reply)

    def test_verification_rejects_invented_unaccepted_or_misquoted_restaurants(self):
        evaluation = self.accept(["DL-017"])
        good = self.generator.generate(self.criteria, self.report, evaluation)
        bad = [
            Recommendation(good.reply + "INVENTED", ("INVENTED",)),
            Recommendation(good.reply, ("DL-027",)),  # retrieved but not accepted
            Recommendation(good.reply.replace("八德路88號", "八德路99號"), ("DL-017",)),
            Recommendation(good.reply, ("DL-017", "DL-017")),
        ]
        for recommendation in bad:
            with self.subTest(ids=recommendation.restaurant_ids):
                self.assertFalse(verify_recommendation(recommendation, self.catalog, evaluation, self.report))
        self.assertFalse(verify_recommendation(good, self.catalog, Evaluation("stop", ("x",)), self.report))


class PipelineRecommendationTests(unittest.TestCase):
    def test_pipeline_returns_three_recommendations_and_keeps_conditions(self):
        pipeline = RecommendationPipeline(model=AcceptModel())
        result = pipeline.handle("A", "想吃麵，不要火鍋")
        self.assertEqual(len(result.restaurant_ids), 3)
        self.assertTrue(set(result.restaurant_ids) <= set(result.evaluation.accepted_ids))
        self.assertTrue(result.reply.startswith("想吃麵食、不要火鍋的話，我推薦這 3 家："))
        self.assertNotIn("目前條件", result.reply)
        follow = pipeline.handle("A", "50元以內")
        self.assertTrue(follow.restaurant_ids)
        self.assertTrue(all(pipeline.catalog[i].price_min_twd <= 50 for i in follow.restaurant_ids))
        self.assertIn("想吃麵食、50 元以內、不要火鍋的話", follow.reply)
        self.assertEqual(follow.criteria.excluded_cuisines, ("火鍋",))

    def test_model_failure_gives_readable_fallback(self):
        from food_crag.adapters.gemini import GeminiError
        result = RecommendationPipeline(model=AcceptModel(GeminiError("network_or_timeout"))).handle("A", "想吃麵")
        self.assertEqual(result.restaurant_ids, ())
        self.assertIn("推薦評估暫時無法完成", result.reply)

    def test_bad_or_crashing_generator_falls_back_to_verified_template(self):
        class Forger:
            def generate(self, criteria, retrieval, evaluation):
                return Recommendation("推薦你 INVENTED 餐廳", ("INVENTED",))

        class Crasher:
            def generate(self, criteria, retrieval, evaluation):
                raise RuntimeError("boom")

        for generator in (Forger(), Crasher()):
            with self.subTest(generator=type(generator).__name__):
                result = RecommendationPipeline(model=AcceptModel(), reply_generator=generator).handle("A", "想吃麵")
                self.assertNotIn("INVENTED", result.restaurant_ids)
                self.assertEqual(len(result.restaurant_ids), 3)
                self.assertNotIn("INVENTED", result.reply)


if __name__ == "__main__":
    unittest.main()
