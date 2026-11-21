"""Dataset integrity and rejection checks using the actual shipped records."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from food_crag.dataset import (
    DEFAULT_DATA_PATH, DatasetError, load_restaurants, read_records, validate_records,
)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.records = read_records()

    def test_demo_size_and_stable_ids(self):
        self.assertGreaterEqual(len(self.records), 30)
        self.assertLessEqual(len(self.records), 50)
        # DL-011 was merged into DL-010 on 2026-11-19; IDs are never reused.
        self.assertEqual({r["id"] for r in self.records}, {f"DL-{i:03}" for i in range(1, 34)} - {"DL-011"})

    def test_draft_is_preserved_and_formal_is_separate(self):
        draft = json.loads(DEFAULT_DATA_PATH.with_name("restaurants.draft.json").read_text(encoding="utf-8"))
        self.assertTrue(all(r["review_status"] == "draft" for r in draft))
        self.assertTrue(all(r["review_status"] == "documented" for r in self.records))

    def test_load_preserves_unknowns_and_provenance(self):
        restaurants = load_restaurants()
        self.assertEqual(len(restaurants), len(self.records))
        for item, raw in zip(restaurants, self.records):
            # Main-dish listed prices come only from cited menu photos.
            self.assertTrue(0 < item.price_min_twd <= item.price_max_twd)
            self.assertTrue(item.price_evidence.startswith("主餐單品標價："))
            self.assertTrue(any("price_evidence" in s["supports"] and s["access_method"] == "manual_entry"
                                and "菜單/" in s["note"] for s in item.sources))
            self.assertIsNone(item.meal_periods)
            self.assertEqual(item.operating_status, "unknown")
            self.assertEqual(item.sources, tuple(raw["sources"]))
            self.assertEqual(item.checked_at, raw["checked_at"])
            self.assertEqual(item.notes, tuple(raw["notes"]))

    def test_original_manual_received_date_is_not_rewritten(self):
        for record in (r for r in self.records if r["id"] in {"DL-031", "DL-032"}):
            source = next(s for s in record["sources"] if s["access_method"] == "manual_entry")
            self.assertEqual(source["reported_at"], "2026-09-07")
            self.assertIsNone(source["checked_at"])
            self.assertIsNone(source["visited_at"])

    def test_bad_records_are_rejected(self):
        changes = {
            "missing field": lambda r: r.pop("address"),
            "empty name": lambda r: r.update(name=" "),
            "bad ID": lambda r: r.update(id="1"),
            "wrong district": lambda r: r.update(district="虎尾鎮"),
            "invalid date": lambda r: r.update(checked_at="2026-02-30"),
            "wrong review date": lambda r: r.update(checked_at="2026-01-01"),
            "boolean price": lambda r: r.update(price_min_twd=True),
            "price without evidence": lambda r: r.update(price_evidence=None),
            "half price range": lambda r: r.update(price_max_twd=None),
            "reversed prices": lambda r: r.update(price_min_twd=200, price_max_twd=100, price_evidence="test"),
            "missing provenance": lambda r: r.update(sources=[]),
            "invalid URL": lambda r: r["source_urls"].append("javascript:alert(1)"),
            "unmatched URL": lambda r: r["source_urls"].append("https://example.com/extra"),
            "unreviewed draft": lambda r: r.update(review_status="draft"),
            "unsupported landmark": lambda r: r.update(landmarks=["火車站"]),
            "unsupported menu": lambda r: (r.update(menu_summary="invented menu"), [s.update(
                supports=[f for f in s["supports"] if f != "menu_summary"]) for s in r["sources"]]),
            "invalid source date": lambda r: r["sources"][0].update(checked_at="yesterday"),
            "invalid source type": lambda r: r["sources"][0].update(source_type=[]),
            "missing support": lambda r: r["sources"].__setitem__(slice(None), [dict(r["sources"][0], supports=["name"])]),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                records = copy.deepcopy(self.records)
                change(records[0])
                with self.assertRaises(DatasetError):
                    validate_records(records)

    def test_duplicate_ids_and_stores_are_rejected(self):
        for field_set in (("id",), ("name", "address")):
            with self.subTest(fields=field_set):
                records = copy.deepcopy(self.records)
                for field in field_set:
                    records[1][field] = records[0][field]
                with self.assertRaises(DatasetError):
                    validate_records(records)

    def test_manual_source_requires_received_date(self):
        record = copy.deepcopy(self.records[-1])
        source = next(s for s in record["sources"] if s["access_method"] == "manual_entry")
        source.pop("reported_at")
        with self.assertRaises(DatasetError):
            validate_records([record])

    def test_manual_source_without_online_review_keeps_null_date(self):
        record = copy.deepcopy(self.records[-1])
        record["sources"] = [s for s in record["sources"] if s["access_method"] == "manual_entry"]
        record["source_urls"] = []
        record["checked_at"] = None
        validate_records([record])

    def test_malformed_json_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("[", encoding="utf-8")
            with self.assertRaises(DatasetError):
                load_restaurants(path)


if __name__ == "__main__":
    unittest.main()
