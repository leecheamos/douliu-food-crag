"""Grounded recommendation replies (step 7).

TemplateReplyGenerator fills a fixed format from the local dataset only. A later
model-written generator must implement the same `generate` interface; its output
goes through the same `verify_recommendation` check before reaching the user.
"""
from collections import Counter
from dataclasses import dataclass

from .retriever import sells


MAX_RECOMMENDATIONS = 3
DISCLAIMER = "※ 價格是菜單上的單品價，不是整餐費用；資料未向店家確認，出發前請再確認營業時間。"


@dataclass(frozen=True)
class Recommendation:
    reply: str
    restaurant_ids: tuple[str, ...] = ()


def _supported(restaurant):
    return {field for source in restaurant.sources for field in source.get("supports", ())}


def _spoken_list(items):
    items = [i for i in items if i]
    return "、".join(items[:-1]) + "和" + items[-1] if len(items) > 1 else "".join(items)


def _wish(criteria):
    """Conversational restatement of the active conditions, e.g. 想吃麵食、100 元以內."""
    parts = []
    if criteria.dishes or criteria.cuisines:
        parts.append("想吃" + "、".join(criteria.dishes or criteria.cuisines))
    if criteria.budget_twd is not None:
        parts.append(f"{criteria.budget_twd} 元以內")
    if criteria.excluded_cuisines:
        parts.append("不要" + "、".join(criteria.excluded_cuisines))
    if criteria.meal:
        parts.append({"lunch": "吃午餐", "dinner": "吃晚餐"}[criteria.meal])
    if criteria.landmark:
        parts.append(f"在{criteria.landmark}附近")
    return "、".join(parts)


def _intro(restaurant, supported, dishes=()):
    hits = [i for i in (restaurant.name, *restaurant.menu_items) if any(d in i for d in dishes)]
    if hits:  # Show the menu items that match the requested dish, e.g. 有賣滷排骨飯.
        return f"{restaurant.name}，有賣{_spoken_list(hits[:2])}" if hits[0] != restaurant.name else restaurant.name
    dishes = [d.strip() for d in restaurant.menu_summary.split("、")] if "menu_summary" in supported else []
    return f"{restaurant.name}，主要賣{_spoken_list(dishes)}" if any(dishes) else restaurant.name


def _price(restaurant):
    low, high = restaurant.price_min_twd, restaurant.price_max_twd
    if not restaurant.price_evidence or "price_evidence" not in _supported(restaurant) or (low is None and high is None):
        return "價格：未知"
    if low is not None and high is not None:
        amount = f"{low} 元" if low == high else f"{low}～{high} 元"
    else:
        amount = f"{low} 元起" if low is not None else f"最高 {high} 元"
    return f"價格：主餐約 {amount}"


def _unknown_summary(retrieval):
    counts = Counter(reason for d in retrieval.decisions if d.outcome == "unknown" for reason in d.reasons)
    return "；".join(f"{reason}（{count} 家）" for reason, count in counts.most_common())


class TemplateReplyGenerator:
    def __init__(self, restaurants):
        self.catalog = {r.id: r for r in restaurants}

    def generate(self, criteria, retrieval, evaluation):
        if evaluation is None or retrieval is None:
            return Recommendation("本輪未執行檢索，沒有推薦。")
        if evaluation.decision == "accept":
            accepted = set(evaluation.accepted_ids)
            picks = [c.restaurant for c in retrieval.candidates if c.restaurant.id in accepted]
            if not criteria.cuisines and not criteria.dishes:  # "都可以": one shop per cuisine for variety.
                seen = set()
                picks = [r for r in picks if r.cuisines[:1] not in seen and not seen.add(r.cuisines[:1])]
            lead = None
            if criteria.dishes:  # Only shops that literally sell the dish, else say so honestly.
                dish = "、".join(criteria.dishes)
                matched = [r for r in picks if any(sells(r, d) for d in criteria.dishes)]
                sellers = {r.id for r in self.catalog.values() if any(sells(r, d) for d in criteria.dishes)}
                similar = "相近的" + ("、".join(criteria.cuisines) or "店家")
                if matched:
                    picks = matched
                elif sellers:
                    reasons = {x for d in retrieval.decisions if d.restaurant_id in sellers for x in d.reasons}
                    why = "（" + "、".join(sorted(reasons)) + "）" if reasons else ""
                    lead = f"有 {len(sellers)} 家賣{dish}，但不符合你的其他條件{why}；先推薦符合條件的{similar}："
                else:
                    lead = f"目前資料裡沒有找到賣{dish}的店，先推薦{similar}："
            picks = [r.id for r in picks][:MAX_RECOMMENDATIONS]
            if not picks:
                return self.unavailable()
            shops = [self.catalog[restaurant_id] for restaurant_id in picks]
            wish = _wish(criteria)
            count = "這家" if len(shops) == 1 else f"這 {len(shops)} 家"
            if lead is None:
                lead = f"{wish}的話，我推薦{count}：" if wish else f"我推薦{count}："
            intro = lead + "；".join(_intro(r, _supported(r), criteria.dishes) for r in shops) + "。"
            blocks = []
            for r in shops:
                lines = [r.name, f"地址：{r.address}", _price(r)]
                if r.map_search_url:
                    lines.append(r.map_search_url)
                blocks.append("\n".join(lines))
            note = DISCLAIMER
            if criteria.meal:  # Service periods are unsourced; meal is only a preference.
                meal = {"lunch": "午餐", "dinner": "晚餐"}[criteria.meal]
                note = f"※ 價格是菜單上的單品價，不是整餐費用；是否有供應{meal}尚未確認，出發前請先查營業時間。"
            return Recommendation("\n\n".join([intro, *blocks, note]), tuple(picks))
        if evaluation.evidence_state == "insufficient":
            detail = _unknown_summary(retrieval)
            return Recommendation(
                "目前資料不足，無法確認有店家符合你的全部條件，所以這次沒有推薦。"
                + (f"不足之處：{detail}。" if detail else "")
                + "我沒有放寬你的條件；如果願意調整，請直接告訴我，例如「取消預算限制」。")
        if evaluation.evidence_state == "no_results":
            return Recommendation("依目前資料，沒有同時符合你條件的店家，所以這次沒有推薦。"
                                  "我沒有放寬你的條件；可以換個料理或取消部分排除條件再試。")
        return self.unavailable()

    @staticmethod
    def unavailable():
        return Recommendation("推薦評估暫時無法完成，請稍後再試一次。"
                              "為避免提供未經評估的結果，這次不列出餐廳。")


def verify_recommendation(recommendation, catalog, evaluation, retrieval):
    """Every recommended restaurant must be real, accepted, retrieved and quoted exactly."""
    ids = recommendation.restaurant_ids
    if not ids:
        return True
    if evaluation is None or evaluation.decision != "accept" or retrieval is None:
        return False
    retrieved = {c.restaurant.id for c in retrieval.candidates}
    if len(ids) > MAX_RECOMMENDATIONS or len(set(ids)) != len(ids):
        return False
    for restaurant_id in ids:
        restaurant = catalog.get(restaurant_id)
        if (restaurant is None or restaurant_id not in evaluation.accepted_ids or restaurant_id not in retrieved
                or restaurant.name not in recommendation.reply or restaurant.address not in recommendation.reply):
            return False
    return True
