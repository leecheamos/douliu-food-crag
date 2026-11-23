"""Item 12 collector: Places responses are faked, nothing calls Google."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from food_crag import collect_places as cp
from food_crag.dataset import load_restaurants, read_records
from food_crag.recommendation import _price


def place(pid="p1", name="阿明雞排", lat=23.7030, lng=120.5388, **extra):
    data = {"id": pid, "displayName": {"text": name}, "formattedAddress": "640台灣雲林縣斗六市中山路999號",
            "location": {"latitude": lat, "longitude": lng}, "types": ["restaurant", "food"],
            "businessStatus": "OPERATIONAL", "userRatingCount": 500, "googleMapsUri": f"https://maps.google.com/?cid={pid}",
            "priceRange": {"startPrice": {"currencyCode": "TWD", "units": "100"},
                           "endPrice": {"currencyCode": "TWD", "units": "200"}},
            "regularOpeningHours": {"periods": [{"open": {"day": d, "hour": 11}, "close": {"day": d, "hour": 21}}
                                                for d in range(7)], "weekdayDescriptions": ["星期一: 11:00–21:00"]}}
    data.update(extra)
    return data


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


class FilterAndParseTests(unittest.TestCase):
    def test_only_open_reviewed_nearby_food_places_are_kept(self):
        self.assertTrue(cp.keep(place()))
        for bad in (place(businessStatus="CLOSED_PERMANENTLY"), place(userRatingCount=3),
                    place(lat=23.75), place(name="清心福全 手搖茶飲"), place(primaryType="cafe"),
                    place(types=["store"]), place(formattedAddress="640台灣雲林縣虎尾鎮中山路1號")):
            with self.subTest(bad=bad):
                self.assertFalse(cp.keep(bad))

    def test_opening_hours_become_meal_periods(self):
        self.assertEqual(cp.meal_periods(place()["regularOpeningHours"]), ["lunch", "dinner"])
        breakfast = {"periods": [{"open": {"day": 1, "hour": 5}, "close": {"day": 1, "hour": 11, "minute": 30}}]}
        self.assertEqual(cp.meal_periods(breakfast), [])
        night = {"periods": [{"open": {"day": 6, "hour": 17}, "close": {"day": 0, "hour": 2}}]}  # Sat 17:00-Sun 02:00
        self.assertEqual(cp.meal_periods(night), ["dinner"])
        self.assertEqual(cp.meal_periods({"periods": [{"open": {"day": 0, "hour": 0}}]}), ["lunch", "dinner"])
        self.assertIsNone(cp.meal_periods(None))

    def test_name_and_type_rules(self):
        self.assertEqual(cp.rule_cuisines(place(name="阿明雞排")), ["炸物"])
        self.assertEqual(cp.rule_cuisines(place(name="老王牛肉麵")), ["牛肉麵"])  # 麵食 fallback dropped.
        self.assertEqual(cp.rule_cuisines(place(name="小李麵店")), ["麵食"])
        self.assertEqual(cp.rule_cuisines(place(name="Sakura", types=["japanese_restaurant"])), ["日式料理"])
        self.assertEqual(cp.rule_cuisines(place(name="好地方")), [])

    def test_address_drops_postcode_and_simplified_county(self):
        for raw in ("640台灣雲林縣斗六市中山路1號", "64072雲林縣斗六市中山路1號", "640云林县斗六市中山路1號"):
            self.assertEqual(cp.address(place(formattedAddress=raw)), "雲林縣斗六市中山路1號")

    def test_price_needs_both_ends_in_twd(self):
        self.assertEqual(cp.price(place()), (100, 200))
        self.assertIsNone(cp.price(place(priceRange={"startPrice": {"currencyCode": "TWD", "units": "100"}})))
        self.assertIsNone(cp.price(place(priceRange=None)))

    def test_duplicates_of_menu_checked_shops_are_flagged(self):
        existing = read_records()
        known = next(r for r in existing if len(r["name"].split()[0]) >= 2)
        self.assertEqual(cp.duplicate_of(place(name=known["name"].split()[0] + " 新分店"), existing), known["id"])
        self.assertEqual(cp.duplicate_of(place(name="完全不同的店"), existing), "")


class FetchTests(unittest.TestCase):
    def test_pages_are_followed_deduplicated_and_capped(self):
        bodies = []

        def opener(request, timeout):
            body = json.loads(request.data)
            bodies.append(body)
            self.assertEqual(request.get_header("X-goog-api-key"), "k")
            self.assertIn("places.priceRange", request.get_header("X-goog-fieldmask"))
            if "pageToken" not in body:
                return Response(json.dumps({"places": [place("a"), place("b", userRatingCount=1)],
                                            "nextPageToken": "t"}).encode())
            return Response(json.dumps({"places": [place("a"), place("c", name="小李麵店")]}).encode())

        places, requests = cp.fetch("k", opener)
        self.assertEqual(requests, len(cp.CENTERS) * len(cp.QUERIES) * 2)
        self.assertEqual(sorted(p["id"] for p in places), ["a", "c"])
        self.assertEqual(bodies[1]["pageToken"], "t")
        with patch.object(cp, "MAX_REQUESTS", 3):
            self.assertEqual(cp.fetch("k", opener)[1], 3)


class ReviewAndBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.paths = patch.multiple(cp, RAW=self.tmp / "raw.json", REVIEW=self.tmp / "review.csv",
                                    OUTPUT=self.tmp / "google.json")
        self.paths.start()
        self.addCleanup(self.paths.stop)
        places = [place("a", "阿明雞排"), place("b", "好地方", formattedAddress="640台灣雲林縣斗六市中山路1000號")]
        cp.RAW.write_text(json.dumps({"fetched_at": "2026-11-21", "places": places}, ensure_ascii=False), encoding="utf-8")

    def test_review_then_build_produces_valid_mergeable_records(self):
        class Model:
            def generate_json(self, payload, schema):
                return {"results": [{"id": "b", "cuisines": ["小吃", "披薩"]}]}  # Unknown labels are dropped.

        rows = cp.review(json.loads(cp.RAW.read_text(encoding="utf-8"))["places"], Model())
        self.assertEqual({r["place_id"]: r["cuisines"] for r in rows}, {"a": "炸物", "b": "小吃"})
        records = cp.build()
        self.assertEqual([r["id"] for r in records], ["DL-101", "DL-102"])
        self.assertEqual(records[0]["meal_periods"], ["lunch", "dinner"])
        merged = load_restaurants(extra=cp.OUTPUT)
        self.assertEqual(len(merged), len(read_records()) + 2)
        shop = next(r for r in merged if r.id == "DL-101")
        self.assertEqual(_price(shop), "價格：約 100～200 元（Google 估計每人消費）")
        from dataclasses import replace
        self.assertEqual(_price(replace(shop, price_min_twd=1)), "價格：約 200 元以內（Google 估計每人消費）")
        from food_crag.domain import SearchCriteria
        from food_crag.retriever import RestaurantRetriever
        bracket = replace(shop, price_min_twd=1)  # Google "up to 200": a 50 budget cannot be confirmed.
        self.assertEqual(RestaurantRetriever.filter(bracket, SearchCriteria(budget_twd=50)).outcome, "unknown")
        self.assertEqual(RestaurantRetriever.filter(bracket, SearchCriteria(budget_twd=200)).outcome, "eligible")

    def test_build_rejects_rows_without_allowed_cuisines(self):
        cp.review(json.loads(cp.RAW.read_text(encoding="utf-8"))["places"], None)
        with cp.REVIEW.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0].update(include="1", cuisines="披薩")
        with cp.REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, cp.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(SystemExit):
            cp.build()


if __name__ == "__main__":
    unittest.main()
