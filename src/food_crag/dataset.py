"""Load source-documented restaurant records, without assuming unknown facts."""
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from .criteria import CUISINES
from .domain import Restaurant

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "restaurants.json"
SOURCE_TYPES = {
    "restaurant_official", "government", "media", "blog", "partner",
    "delivery_platform", "corporate_document", "personal_recommendation",
    "recruitment_platform", "business_directory",
}
REQUIRED = {
    "id", "name", "county", "district", "address", "source_urls", "checked_at",
    "cuisines", "price_min_twd", "price_max_twd", "price_evidence",
    "menu_summary", "menu_items", "opening_hours", "landmarks", "sources", "notes",
    "map_search_url", "meal_periods", "operating_status", "review_status",
}


class DatasetError(ValueError):
    """Invalid data must not silently enter downstream processing."""


def _require(condition, message):
    if not condition:
        raise DatasetError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _strings(value):
    return isinstance(value, list) and all(_text(item) for item in value)


def _date(value, label, nullable=False):
    if nullable and value is None:
        return
    _require(isinstance(value, str), f"{label}: expected ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise DatasetError(f"{label}: invalid date") from error
    _require(parsed.isoformat() == value, f"{label}: expected YYYY-MM-DD")


def _url(value, label):
    _require(_text(value), f"{label}: expected URL")
    try:
        parts = urlsplit(value)
        valid = parts.scheme in {"http", "https"} and bool(parts.hostname)
        valid = valid and parts.username is None and parts.password is None
    except ValueError:
        valid = False
    _require(valid, f"{label}: expected HTTP(S) URL without credentials")


def validate_records(records):
    """Validate structure/provenance; this does not confirm current store facts."""
    _require(isinstance(records, list) and records, "dataset: expected nonempty list")
    ids, stores = set(), set()
    for index, record in enumerate(records):
        label = f"record[{index}]"
        _require(isinstance(record, dict), f"{label}: expected object")
        _require(REQUIRED <= record.keys(), f"{label}: missing required fields")
        _require(_text(record["id"]) and re.fullmatch(r"DL-\d{3}", record["id"]), f"{label}: invalid ID")
        label = record["id"]
        _require(label not in ids, f"{label}: duplicate ID")
        ids.add(label)
        for field in ("name", "county", "district", "address"):
            _require(_text(record[field]), f"{label}.{field}: required text")
        _require(record["county"] == "雲林縣" and record["district"] == "斗六市", f"{label}: outside dataset scope")
        _require("斗六市" in record["address"], f"{label}: address outside scope")
        store = (re.sub(r"\s+", "", record["name"]), re.sub(r"\s+", "", record["address"]))
        _require(store not in stores, f"{label}: duplicate name/address")
        stores.add(store)
        for field in ("source_urls", "cuisines", "landmarks", "notes"):
            _require(_strings(record[field]), f"{label}.{field}: expected text list")
        _require(bool(record["cuisines"]), f"{label}: missing cuisine classification")
        _require(len(record["source_urls"]) == len(set(record["source_urls"])), f"{label}: duplicate source URL")
        for url in record["source_urls"]:
            _url(url, label)
        _url(record["map_search_url"], f"{label}.map_search_url")
        _date(record["checked_at"], f"{label}.checked_at", nullable=True)
        _require(isinstance(record["menu_summary"], str), f"{label}: invalid menu summary")
        items = record["menu_items"]
        _require(isinstance(items, list) and all(_text(i) for i in items) and len(items) == len(set(items)),
                 f"{label}: menu_items must be unique non-empty strings")
        prices = record.get("menu_prices", {})  # Optional: Google records have no menu.
        _require(isinstance(prices, dict) and set(prices) <= set(items)
                 and all(type(v) is int and v > 0 for v in prices.values()),
                 f"{label}: menu_prices must map menu_items to positive integers")
        tags = record.get("item_tags", {})  # Optional hand labels for names keywords misread.
        _require(isinstance(tags, dict) and set(tags) <= set(items)
                 and all(isinstance(v, list) and set(v) <= set(CUISINES) for v in tags.values()),
                 f"{label}: item_tags must map menu_items to known cuisines")
        for field in ("opening_hours", "price_evidence"):
            _require(record[field] is None or _text(record[field]), f"{label}.{field}: expected text or null")
        for field in ("price_min_twd", "price_max_twd"):
            value = record[field]
            _require(value is None or (type(value) is int and value > 0), f"{label}.{field}: expected positive integer or null")
        low, high = record["price_min_twd"], record["price_max_twd"]
        _require(low is None or high is None or low <= high, f"{label}: reversed price range")
        if low is not None or high is not None:
            _require(low is not None and high is not None, f"{label}: price range needs both ends")
            _require(_text(record["price_evidence"]), f"{label}: price requires evidence")
        periods = record["meal_periods"]
        _require(periods is None or (_strings(periods) and set(periods) <= {"lunch", "dinner"}), f"{label}: invalid meal periods")
        _require(record["operating_status"] == "unknown", f"{label}: current operation not verified in this dataset")
        _require(record["review_status"] == "documented", f"{label}: record not source-documented")
        sources = record["sources"]
        _require(isinstance(sources, list) and sources, f"{label}: missing provenance")
        urls, checked, supported = set(), [], set()
        for source in sources:
            _require(isinstance(source, dict), f"{label}: invalid source")
            keys = {"publisher", "url", "source_type", "checked_at", "access_method", "source_date", "source_date_kind", "supports"}
            _require(keys <= source.keys(), f"{label}: incomplete source")
            _require(_text(source["publisher"]), f"{label}: missing publisher")
            _require(_text(source["source_type"]) and source["source_type"] in SOURCE_TYPES, f"{label}: invalid source type")
            method = source["access_method"]
            _require(_text(method) and method in {"page", "search_index", "manual_entry", "api"}, f"{label}: invalid access method")
            _require(_strings(source["supports"]) and source["supports"], f"{label}: missing supported fields")
            supported.update(source["supports"])
            _date(source["checked_at"], f"{label}.source.checked_at", nullable=method == "manual_entry")
            _date(source["source_date"], f"{label}.source.source_date", nullable=True)
            _require(source["source_date_kind"] is None or (_text(source["source_date_kind"]) and source["source_date_kind"] in {"published", "updated", "published_or_updated"}), f"{label}: invalid source date kind")
            if source["source_date"] is not None:
                _require(source["source_date_kind"] is not None, f"{label}: missing source date kind")
            if method == "manual_entry":
                _require(source["source_type"] == "personal_recommendation", f"{label}: manual source type mismatch")
                _date(source.get("reported_at"), f"{label}.reported_at")
                _date(source.get("visited_at"), f"{label}.visited_at", nullable=True)
                _require(_text(source.get("note")), f"{label}: missing manual source note")
            else:
                _require(source["url"] is not None, f"{label}: online source missing URL")
            if source["url"] is not None:
                _url(source["url"], label)
                urls.add(source["url"])
            if source["checked_at"] is not None:
                checked.append(source["checked_at"])
        _require(urls == set(record["source_urls"]), f"{label}: source URLs do not match provenance")
        _require({"name", "address", "cuisines"} <= supported, f"{label}: missing name/address/cuisine provenance")
        _require(record["checked_at"] == (max(checked) if checked else None), f"{label}: checked_at must match latest source review")
        if low is not None or high is not None or prices:
            _require("price_evidence" in supported, f"{label}: price missing source support")
        if record["landmarks"]:
            _require("landmarks" in supported, f"{label}: landmarks missing source support")
        for field in ("opening_hours", "menu_summary", "menu_items", "meal_periods"):
            if record[field]:
                _require(field in supported, f"{label}: {field} missing source support")
    return records


def read_records(path=DEFAULT_DATA_PATH, extra=None):
    """extra: optional second file (the gitignored Google Places records), validated together."""
    records = []
    for file in (path, extra):
        if file is None:
            continue
        file = Path(file) if Path(file).is_absolute() else PROJECT_ROOT / file
        try:
            records += json.loads(file.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            raise DatasetError(f"Cannot read restaurant dataset: {error}") from error
    return validate_records(records)


def load_restaurants(path=DEFAULT_DATA_PATH, extra=None):
    """Explicitly map metadata and lists; never pass arbitrary JSON kwargs."""
    result = []
    for record in read_records(path, extra):
        fields = {key: record[key] for key in Restaurant.__dataclass_fields__ if key not in ("menu_prices", "item_tags")}
        fields["menu_prices"] = tuple(record.get("menu_prices", {}).items())
        fields["item_tags"] = tuple((item, tuple(tags)) for item, tags in record.get("item_tags", {}).items())
        for key in ("source_urls", "cuisines", "menu_items", "landmarks", "sources", "notes"):
            fields[key] = tuple(fields[key])
        if fields["meal_periods"] is not None:
            fields["meal_periods"] = tuple(fields["meal_periods"])
        result.append(Restaurant(**fields))
    return result


def summary(records):
    return {
        "records": len(records),
        "with_price": sum(r["price_min_twd"] is not None or r["price_max_twd"] is not None for r in records),
        "with_menu": sum(bool(r["menu_summary"]) for r in records),
        "with_hours": sum(r["opening_hours"] is not None for r in records),
        "with_landmarks": sum(bool(r["landmarks"]) for r in records),
        "access_methods": dict(Counter(s["access_method"] for r in records for s in r["sources"])),
        "current_operation_verified": 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    try:
        print(json.dumps(summary(read_records(args.path)), indent=2))
    except DatasetError as error:
        parser.exit(1, f"Dataset error: {error}\n")
