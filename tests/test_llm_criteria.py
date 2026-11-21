"""Free-form fallback parser: model output is data and is validated locally."""
import unittest

from food_crag.domain import SearchCriteria
from food_crag.llm_criteria import GeminiCriteriaParser, to_criteria
from food_crag.pipeline import RecommendationPipeline


def output(**changes):
    base = dict(action="update", cuisines=[], excluded_cuisines=[], dishes=[], budget_twd=0,
                meal="any", landmark="none", question="")
    base.update(changes)
    return base


class FakeParserModel:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def generate_json(self, payload, schema):
        self.calls.append(payload)
        if self.error:
            raise self.error
        return self.result


class AcceptModel:
    def generate_json(self, payload, schema):
        return dict(decision="accept", evidence_state="sufficient", rewrite_query="", reasons=["測試"],
                    accepted_ids=[c["restaurant"]["id"] for c in payload["candidates"]])


class ValidationTests(unittest.TestCase):
    def test_free_form_dish_request_becomes_category_plus_dish(self):
        result = to_criteria(output(cuisines=["飯食"], dishes=["豬排飯"]), "豬排飯", SearchCriteria())
        self.assertEqual(result.criteria, SearchCriteria(cuisines=("飯食",), dishes=("豬排飯",)))
        self.assertEqual(result.questions, ())

    def test_previous_conditions_can_be_carried_over(self):
        previous = SearchCriteria(budget_twd=100, cuisines=("飯食",), dishes=("豬排飯",))
        result = to_criteria(output(cuisines=["麵食"], budget_twd=100), "麵的話可以有什麼", previous)
        self.assertEqual(result.criteria, SearchCriteria(budget_twd=100, cuisines=("麵食",)))

    def test_invented_values_are_rejected(self):
        message = "想吃點好吃的"
        for bad in [output(cuisines=["麵食"], dishes=["豬排飯"]),       # dish not in message
                    output(cuisines=["麵食"], budget_twd=150),          # budget number not in message
                    output(cuisines=["披薩"]),                           # outside vocabulary
                    output(cuisines=["麵食"], landmark="台北101"),
                    output(cuisines=["麵食"], meal="breakfast"),
                    output(cuisines=["麵食", "麵食"]),
                    output(cuisines=["麵食"], budget_twd=-5),
                    output(cuisines=["麵食"], budget_twd=True),
                    output(action="other"),
                    output(),                                          # nothing understood
                    {"action": "update"}, "not a dict"]:
            with self.subTest(bad=bad):
                self.assertIsNone(to_criteria(bad, message, SearchCriteria()))

    def test_budget_and_dish_from_the_message_are_accepted(self):
        result = to_criteria(output(cuisines=["麵食"], dishes=["牛肉麵"], budget_twd=150),
                             "想吃牛肉麵 150 塊左右", SearchCriteria())
        self.assertEqual(result.criteria, SearchCriteria(budget_twd=150, cuisines=("麵食",), dishes=("牛肉麵",)))

    def test_chinese_numeral_budgets_count_as_written_by_the_user(self):
        from food_crag.llm_criteria import _chinese_number
        for word, value in [("一百", 100), ("一百五", 150), ("兩百五十", 250), ("三千", 3000), ("八十", 80), ("十五", 15)]:
            self.assertEqual(_chinese_number(word), value, word)
        result = to_criteria(output(cuisines=["麵食"], budget_twd=100), "有沒有便宜一點的麵，一百塊以內", SearchCriteria())
        self.assertEqual(result.criteria.budget_twd, 100)
        self.assertIsNone(to_criteria(output(cuisines=["麵食"], budget_twd=200), "一百塊以內的麵", SearchCriteria()))

    def test_clarify_reset_and_conflict(self):
        previous = SearchCriteria(cuisines=("麵食",))
        clarify = to_criteria(output(action="clarify", question="目前無法確認過敏原，請改用料理或預算描述。"), "我花生過敏", previous)
        self.assertEqual(clarify.criteria, previous)
        self.assertEqual(len(clarify.questions), 1)
        self.assertIsNone(to_criteria(output(action="clarify", question=" "), "x", previous))
        self.assertTrue(to_criteria(output(action="reset"), "重來", previous).reset)
        conflict = to_criteria(output(cuisines=["牛肉麵"], excluded_cuisines=["麵食"]), "想吃牛肉麵但不要麵", previous)
        self.assertEqual(conflict.criteria, previous)
        self.assertIn("衝突", conflict.questions[0])

    def test_guide_action_keeps_conditions_and_flags_guidance(self):
        previous = SearchCriteria(budget_twd=100)
        result = to_criteria(output(action="guide", question="辛苦了！要不要吃點熱的？想吃飯還是麵？"), "今天好累", previous)
        self.assertTrue(result.guide)
        self.assertEqual(result.criteria, previous)
        self.assertIsNone(to_criteria(output(action="guide", question=""), "今天好累", previous))

    def test_api_failure_returns_none(self):
        parser = GeminiCriteriaParser(FakeParserModel(error=RuntimeError("network_or_timeout")))
        self.assertIsNone(parser.parse("豬排飯", SearchCriteria()))


class PipelineFallbackTests(unittest.TestCase):
    def test_rules_first_model_only_for_unreadable_messages(self):
        model = FakeParserModel(output(cuisines=["飯食"], dishes=["雞排飯"]))
        pipeline = RecommendationPipeline(model=AcceptModel(), parser_model=model)
        readable = pipeline.handle("A", "想吃麵，100元以內")
        self.assertEqual(model.calls, [])  # The rules understood it; no model call.
        self.assertEqual(readable.criteria.budget_twd, 100)
        free = pipeline.handle("A", "雞排飯")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["previous"]["budget_twd"], 100)
        self.assertEqual(free.criteria.dishes, ("雞排飯",))
        # No shop's menu lists 雞排飯, so the reply says so instead of implying it.
        self.assertTrue(free.reply.startswith("目前資料裡沒有找到賣雞排飯的店，先推薦相近的飯食："))
        # The fixed output format is unchanged.
        self.assertIn("地址：", free.reply)
        self.assertIn("價格：主餐約", free.reply)

    def test_fixed_guidance_with_buttons_when_model_fails_or_is_unsure(self):
        from food_crag.pipeline import GUIDE_TEXT
        for model in (FakeParserModel(error=RuntimeError("boom")), FakeParserModel(output())):
            with self.subTest(model=model.result):
                pipeline = RecommendationPipeline(model=AcceptModel(), parser_model=model)
                result = pipeline.handle("A", "豬排飯")
                self.assertEqual(result.status, "guidance")
                self.assertEqual(result.reply, GUIDE_TEXT)
                self.assertEqual(len(result.quick_replies), 4)
                self.assertEqual(result.quick_replies[-1], "都可以")
                self.assertIsNone(pipeline.store.get("A"))

    def test_model_guidance_is_used_unless_it_names_a_shop(self):
        from food_crag.pipeline import GUIDE_TEXT
        pipeline = RecommendationPipeline(model=AcceptModel(), parser_model=FakeParserModel(
            output(action="guide", question="辛苦了！想吃飯還是麵？")))
        self.assertEqual(pipeline.handle("A", "今天好累").reply, "辛苦了！想吃飯還是麵？")
        shop = pipeline.retriever.restaurants[0].name
        pipeline.free_parser.model.result = output(action="guide", question=f"去{shop}吧！")
        self.assertEqual(pipeline.handle("A", "不知道吃什麼").reply, GUIDE_TEXT)

    def test_guide_buttons_are_understood_and_lead_to_budget_buttons(self):
        model = FakeParserModel(error=RuntimeError("must not be called"))
        pipeline = RecommendationPipeline(model=AcceptModel(), parser_model=model)
        for button in pipeline.guide_buttons:
            with self.subTest(button=button):
                result = pipeline.handle("B", button)
                self.assertEqual(result.status, "criteria_updated")
                self.assertEqual(result.quick_replies, ("100元以內", "200元以內", "不限預算"))
        self.assertEqual(model.calls, [])
        anything = pipeline.handle("B", "都可以")
        cuisines = [pipeline.catalog[i].cuisines[:1] for i in anything.restaurant_ids]
        self.assertEqual(len(cuisines), 3)
        self.assertEqual(len(set(cuisines)), 3)  # Variety when no cuisine is set.
        self.assertEqual(pipeline.handle("B", "200元以內").quick_replies, ())

    def test_only_shops_that_sell_the_dish_are_recommended(self):
        model = FakeParserModel(output(cuisines=["麵食"], dishes=["牛肉麵"]))
        result = RecommendationPipeline(model=AcceptModel(), parser_model=model).handle("A", "牛肉麵好了")
        self.assertTrue(result.reply.startswith("想吃牛肉麵的話"))
        self.assertTrue(result.restaurant_ids)
        from food_crag.retriever import sells
        catalog = RecommendationPipeline(model=AcceptModel()).catalog
        for i in result.restaurant_ids:
            self.assertTrue(sells(catalog[i], "牛肉麵"), i)

    def test_sellers_blocked_by_budget_are_named_not_recommended(self):
        # Both beef-noodle shops' cheapest mains exceed 35; other noodle shops start at 30.
        model = FakeParserModel(output(cuisines=["麵食"], dishes=["牛肉麵"], budget_twd=35))
        result = RecommendationPipeline(model=AcceptModel(), parser_model=model).handle("A", "牛肉麵 35 塊左右")
        self.assertRegex(result.reply, r"^有 \d+ 家賣牛肉麵，但不符合你的其他條件（[^）]*預算[^）]*）；先推薦符合條件的相近的麵食：")

    def test_sells_matches_main_part_and_staple(self):
        from food_crag.retriever import sells
        from food_crag.domain import Restaurant
        def shop(*items, summary=""):
            return Restaurant("X", "測試", "雲林縣", "斗六市", "地址", (), None, menu_summary=summary, menu_items=items)
        self.assertTrue(sells(shop("厚切豬排飯"), "豬排"))
        self.assertTrue(sells(shop("咖哩豬排蛋包飯"), "豬排飯"))
        self.assertTrue(sells(shop("咖哩豬排蛋包飯（大）"), "豬排飯"))
        self.assertTrue(sells(shop("牛肉乾拌麵"), "牛肉麵"))
        self.assertFalse(sells(shop("排骨飯"), "豬排飯"))       # no 豬排
        self.assertFalse(sells(shop("濃醬咖哩炸飯糰"), "咖哩飯"))  # ends in 糰, not 飯
        self.assertFalse(sells(shop("飯糰豬排"), "豬排飯"))     # staple before main part
        self.assertFalse(sells(shop("蝦仁炒麵"), "炒飯"))       # one-character main part stays literal
        self.assertFalse(sells(shop(summary="豬排、白飯"), "豬排飯"))  # summary checked per item

    def test_dish_with_budget_says_budget_is_per_shop(self):
        model = FakeParserModel(output(cuisines=["麵食"], dishes=["牛肉麵"], budget_twd=150))
        result = RecommendationPipeline(model=AcceptModel(), parser_model=model).handle("A", "牛肉麵 150 塊左右")
        self.assertTrue(result.restaurant_ids)
        self.assertIn("預算是用店內最便宜的主餐判斷，不一定是牛肉麵的價格", result.reply)

    def test_rule_dish_names_reset_when_food_wish_changes(self):
        model = FakeParserModel(output(cuisines=["飯食"], dishes=["豬排飯"]))
        pipeline = RecommendationPipeline(model=AcceptModel(), parser_model=model)
        pipeline.handle("A", "豬排飯")
        self.assertEqual(pipeline.handle("A", "想吃麵").criteria.dishes, ())
        pipeline.handle("A", "豬排飯")
        self.assertEqual(pipeline.handle("A", "200元以內").criteria.dishes, ("豬排飯",))


if __name__ == "__main__":
    unittest.main()
