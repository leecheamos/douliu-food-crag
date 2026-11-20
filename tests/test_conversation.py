"""Behavior cases for supported language, rollback and real HTTP conversations."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import threading
import unittest
from urllib.request import Request, urlopen

from food_crag.criteria import RuleCriteriaParser
from food_crag.domain import SearchCriteria
from food_crag.pipeline import RecommendationPipeline
from food_crag.server import create_server


class CriteriaTests(unittest.TestCase):
    def setUp(self):
        self.parser = RuleCriteriaParser()

    def test_common_phrases(self):
        cases = [
            ("想吃麵", {"cuisines": ("麵食",)}),
            ("想吃麵類", {"cuisines": ("麵食",)}),
            ("２００元以內，不要火鍋", {"budget_twd": 200, "excluded_cuisines": ("火鍋",)}),
            ("預算改成150", {"budget_twd": 150}),
            ("預算不超過200元", {"budget_twd": 200}),
            ("最多200元", {"budget_twd": 200}),
            ("午餐想吃牛肉麵", {"meal": "lunch", "cuisines": ("牛肉麵",)}),
            ("晚餐想吃飯", {"meal": "dinner", "cuisines": ("飯食",)}),
            ("想吃早午餐", {"meal": None, "cuisines": ("早午餐",)}),
            ("在斗六市，雲科大附近吃麵", {"landmark": "雲林科技大學", "cuisines": ("麵食",)}),
            ("斗六車站附近想吃火鍋", {"landmark": "斗六車站", "cuisines": ("火鍋",)}),
            ("不要火鍋和牛排", {"excluded_cuisines": ("火鍋", "牛排")}),
            ("不要 火鍋 和 牛排", {"excluded_cuisines": ("火鍋", "牛排")}),
            ("麵或飯", {"cuisines": ("麵食", "飯食")}),
            ("不要牛肉麵，想吃麵", {"cuisines": ("麵食",), "excluded_cuisines": ("牛肉麵",)}),
        ]
        for message, expected in cases:
            with self.subTest(message=message):
                result = self.parser.parse(message)
                self.assertEqual(result.questions, ())
                for field, value in expected.items():
                    self.assertEqual(getattr(result.criteria, field), value)

    def test_unknown_or_ambiguous_turns_preserve_every_previous_field(self):
        previous = SearchCriteria(budget_twd=200, meal="lunch", cuisines=("麵食",),
                                  excluded_cuisines=("火鍋",), landmark="斗六車站")
        messages = [
            "想吃火鍋", "不要麵", "預算150元，想吃火鍋", "100到200元",
            "200元以上", "不到200元", "100.5元", "預算-100元", "預算0元",
            "預算100元或200元", "兩百元以內", "預算" + "9" * 5000,
            "午餐和晚餐", "雲科大和斗六車站附近", "台北吃麵", "虎尾鎮吃麵",
            "200元以內，花生過敏", "不要不吃火鍋", "不是不吃火鍋",
            "不想吃火鍋也可以", "不吃晚餐", "不要在雲科大附近", "走路5分鐘內",
            "素食", "外帶", "不要太辣", "重設後想吃麵", "嗨",
            "不要火鍋，牛排", "不要麵，牛肉麵也可以",
        ]
        for message in messages:
            with self.subTest(message=message[:60]):
                result = self.parser.parse(message, previous)
                self.assertTrue(result.questions)
                self.assertEqual(result.criteria, previous)
                self.assertFalse(result.reset)

    def test_no_landmark_cannot_resolve_nearby(self):
        result = self.parser.parse("附近吃麵")
        self.assertTrue(result.questions)
        self.assertEqual(result.criteria, SearchCriteria())

    def test_broad_exclusion_conflicts_with_specific_preference(self):
        result = self.parser.parse("想吃牛肉麵，不要麵")
        self.assertTrue(result.questions)

    def test_clear_each_field_preserves_others(self):
        previous = SearchCriteria(budget_twd=200, meal="lunch", cuisines=("麵食",),
                                  excluded_cuisines=("火鍋", "牛排"), landmark="斗六車站")
        for message, field, value in [
            ("取消預算限制", "budget_twd", None),
            ("取消料理限制", "cuisines", ()),
            ("取消排除條件", "excluded_cuisines", ()),
            ("取消地標限制", "landmark", None),
            ("時段不限", "meal", None),
        ]:
            with self.subTest(message=message):
                result = self.parser.parse(message, previous)
                expected = asdict(previous)
                expected[field] = value
                self.assertEqual(result.questions, ())
                self.assertEqual(asdict(result.criteria), expected)

    def test_cancel_one_exclusion_does_not_clear_other_exclusions(self):
        previous = SearchCriteria(excluded_cuisines=("火鍋", "牛排"))
        for message in ("取消排除火鍋", "取消排除 火鍋", "火鍋也可以"):
            with self.subTest(message=message):
                result = self.parser.parse(message, previous)
                self.assertEqual(result.questions, ())
                self.assertEqual(result.criteria.excluded_cuisines, ("牛排",))
                self.assertEqual(result.criteria.cuisines, ())

    def test_new_positive_preference_replaces_and_explicit_addition_appends(self):
        previous = SearchCriteria(cuisines=("麵食",), budget_twd=200)
        result = self.parser.parse("想吃飯", previous)
        self.assertEqual(result.criteria.cuisines, ("飯食",))
        self.assertEqual(result.criteria.budget_twd, 200)
        result = self.parser.parse("也想吃飯", previous)
        self.assertEqual(result.criteria.cuisines, ("麵食", "飯食"))
        result = self.parser.parse("想吃火鍋，也想吃飯", previous)
        self.assertEqual(result.criteria.cuisines, ("火鍋", "飯食"))


class ConversationTests(unittest.TestCase):
    def test_required_multiturn_sequence_and_reset(self):
        pipeline = RecommendationPipeline()
        result = pipeline.handle("A", "200元以內，不要火鍋")
        self.assertEqual(result.status, "criteria_updated")
        result = pipeline.handle("A", "想吃麵")
        self.assertEqual(result.criteria, SearchCriteria(budget_twd=200, cuisines=("麵食",), excluded_cuisines=("火鍋",)))
        result = pipeline.handle("A", "預算改成150")
        self.assertEqual(result.criteria.budget_twd, 150)
        self.assertEqual(result.criteria.cuisines, ("麵食",))
        result = pipeline.handle("A", "重設")
        self.assertEqual(result.status, "reset")
        self.assertIsNone(pipeline.store.get("A"))
        result = pipeline.handle("A", "想吃飯")
        self.assertEqual(result.criteria, SearchCriteria(cuisines=("飯食",)))

    def test_users_are_isolated_and_reset_does_not_affect_other_user(self):
        pipeline = RecommendationPipeline()
        pipeline.handle("A", "100元以內，不要火鍋")
        pipeline.handle("B", "300元以內，想吃火鍋")
        pipeline.handle("A", "重設")
        self.assertEqual(pipeline.store.get("B"), SearchCriteria(budget_twd=300, cuisines=("火鍋",)))
        self.assertEqual(RecommendationPipeline().store.get("B"), None)

    def test_clarification_does_not_change_stored_state(self):
        pipeline = RecommendationPipeline()
        saved = pipeline.handle("A", "200元以內，不要火鍋").criteria
        rejected = pipeline.handle("A", "150元以內，想吃火鍋")
        self.assertEqual(rejected.status, "clarification_required")
        self.assertEqual(pipeline.store.get("A"), saved)
        accepted = pipeline.handle("A", "取消排除火鍋，想吃火鍋")
        self.assertEqual(accepted.status, "criteria_updated")
        self.assertEqual(accepted.criteria.excluded_cuisines, ())
        self.assertEqual(accepted.restaurant_ids, ())

    def test_concurrent_updates_do_not_lose_independent_fields(self):
        pipeline = RecommendationPipeline()
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda message: pipeline.handle("A", message),
                                        ["200元以內", "不要火鍋", "想吃麵"]))
        self.assertTrue(all(r.status == "criteria_updated" for r in results))
        self.assertEqual(pipeline.store.get("A"), SearchCriteria(budget_twd=200, cuisines=("麵食",), excluded_cuisines=("火鍋",)))


class HTTPConversationTests(unittest.TestCase):
    def test_real_http_multiturn_isolation_clarification_and_reset(self):
        server = create_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def send(user, message):
            request = Request(
                f"http://127.0.0.1:{server.server_port}/api/chat",
                data=json.dumps({"user_id": user, "message": message}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=3) as response:
                self.assertEqual(response.status, 200)
                return json.load(response)

        try:
            send("A", "200元以內，不要火鍋")
            a = send("A", "想吃麵")
            self.assertEqual(a["criteria"]["budget_twd"], 200)
            self.assertEqual(a["criteria"]["excluded_cuisines"], ["火鍋"])
            self.assertEqual(a["criteria"]["cuisines"], ["麵食"])
            b = send("B", "想吃火鍋")
            self.assertIsNone(b["criteria"]["budget_twd"])
            self.assertEqual(b["criteria"]["excluded_cuisines"], [])
            conflict = send("A", "預算150元，想吃火鍋")
            self.assertEqual(conflict["status"], "clarification_required")
            self.assertEqual(conflict["criteria"], a["criteria"])
            self.assertTrue(conflict["questions"])
            reset = send("A", "重設")
            self.assertEqual(reset["status"], "reset")
            self.assertIsNone(reset["criteria"]["budget_twd"])
            self.assertEqual(send("B", "晚餐")["criteria"]["cuisines"], ["火鍋"])
            self.assertEqual(reset["restaurant_ids"], [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
