"""Grounded recommendation replies (step 7).

TemplateReplyGenerator fills a fixed format from the local dataset only. A later
model-written generator must implement the same `generate` interface; its output
goes through the same `verify_recommendation` check before reaching the user.
"""
from dataclasses import dataclass

from .criteria import ALIASES, PARENTS
from .retriever import (BUDGET_UNKNOWN, OVER_BUDGET, ROUGH_PRICE, cuisine_labels, fitting_items, has_dish,
                        item_labels, sells)


MAX_RECOMMENDATIONS = 3
MENU_PRICE_NOTE = "※ 價格是菜單上的單品價，不是整餐費用；"
GOOGLE_PRICE_NOTE = "※ 菜單價是單品價、Google 價位是每人消費估計，都不是整餐費用；"
DISCLAIMER = MENU_PRICE_NOTE + "資料未向店家確認，出發前請再確認營業時間。"
GOOGLE = "Google Places API"
DROP_BUDGET = "取消預算限制"
MORE_BUTTON = "還有其他的嗎？"
NOT_RELAXED = "所以這次先不推薦，也沒有自己放寬你的條件。"


@dataclass(frozen=True)
class Recommendation:
    reply: str
    restaurant_ids: tuple[str, ...] = ()
    quick_replies: tuple[str, ...] = ()  # Next steps offered when nothing is recommended.


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


def _intro(restaurant, supported, criteria):
    """One numbered line. Returns (text, priced): priced means the line lists in-budget item prices."""
    if criteria.budget_twd is not None and "price_evidence" in supported:
        within = [f"{i} {p} 元" for i, p in fitting_items(restaurant, criteria) or () if p <= criteria.budget_twd]
        if within:
            return f"{restaurant.name}，{criteria.budget_twd} 元以內有：{'、'.join(within[:3])}", True
    dishes = criteria.dishes
    hits = [i for i in (restaurant.name, *restaurant.menu_items) if any(has_dish(restaurant, d, i) for d in dishes)]
    requested = {ALIASES.get(c, c) for c in criteria.cuisines}
    if not hits and requested and not requested & cuisine_labels(restaurant.cuisines):
        # Found through its menu (e.g. 唐揚雞 at a Japanese shop for 炸物): show those items.
        hits = [i for i in restaurant.menu_items if requested & item_labels(restaurant, i)]
    if hits:  # Show the menu items that match the requested dish, e.g. 有賣：滷排骨飯.
        return (f"{restaurant.name}，有賣：{_spoken_list(hits[:2])}" if hits[0] != restaurant.name else restaurant.name), False
    summary = [d.strip() for d in restaurant.menu_summary.split("、")] if "menu_summary" in supported else []
    return (f"{restaurant.name}，主要賣：{_spoken_list(summary)}" if any(summary) else restaurant.name), False


def _price(restaurant):
    low, high = restaurant.price_min_twd, restaurant.price_max_twd
    if not restaurant.price_evidence or "price_evidence" not in _supported(restaurant) or (low is None and high is None):
        return "價格：未知"
    if low is not None and high is not None:
        amount = f"{low} 元" if low == high else f"{low}～{high} 元"
    else:
        amount = f"{low} 元起" if low is not None else f"最高 {high} 元"
    if _google_price(restaurant):
        if low is not None and low <= 1:  # Google's lowest bracket starts at 1: it means "up to N".
            amount = f"{high} 元以內"
        return f"價格：約 {amount}（Google 估計每人消費）"
    return f"價格：主餐約 {amount}"


def _google_price(restaurant):
    return any(s.get("publisher") == GOOGLE and "price_evidence" in s.get("supports", ()) for s in restaurant.sources)


def _no_shop_reply(criteria, retrieval, evaluation):
    """Conversational no-recommendation reply (user decision 2026-11-23); counts come from the decisions."""
    kind = "、".join(criteria.dishes or criteria.cuisines)
    budget = criteria.budget_twd
    if evaluation.evidence_state == "insufficient":
        unknown = [d for d in retrieval.decisions if d.outcome == "unknown"]
        priced_out = [d for d in unknown if set(d.reasons) <= BUDGET_UNKNOWN]
        if priced_out and budget is not None:
            rough = all(d.reasons == (ROUGH_PRICE,) for d in priced_out)
            why = "只有 Google 的粗略價位" if rough else "沒有確切的標價"
            return Recommendation(
                f"{kind or '符合條件的店'}有 {len(priced_out)} 家，不過你目前的預算還是 {budget} 元以內，"
                f"這些店{why}，沒辦法確定 {budget} 元吃得到。要不要取消預算限制，我再幫你找？",
                quick_replies=(DROP_BUDGET, MORE_BUTTON))
        if any("料理細分類資料不足" in d.reasons for d in unknown):
            parent = next((PARENTS[c] for c in criteria.cuisines if c in PARENTS), None)
            offer = f"要不要換成{parent}，或看看其他類型？" if parent else "要不要看看其他類型？"
            return Recommendation(f"資料裡還沒有店被分到{kind}，我先不亂推薦。{offer}",
                                  quick_replies=(*((f"想吃{parent}",) if parent else ()), MORE_BUTTON))
        reasons = "、".join(sorted({r for d in unknown for r in d.reasons}))
        return Recommendation(f"有 {len(unknown)} 家店的資料不夠，沒辦法確認是否符合你的條件（{reasons}），"
                              + NOT_RELAXED + "要不要調整條件，或看看其他類型？", quick_replies=(MORE_BUTTON,))
    over = [d for d in retrieval.decisions if d.outcome == "excluded" and set(d.reasons) <= OVER_BUDGET]
    if over and budget is not None:
        return Recommendation(f"{kind or '符合條件'}的店有 {len(over)} 家，但價格都超過 {budget} 元，"
                              + NOT_RELAXED + "要不要取消預算限制，或換個類型？",
                              quick_replies=(DROP_BUDGET, MORE_BUTTON))
    return Recommendation(f"依目前資料，沒有符合「{_wish(criteria) or '你的條件'}」的店，"
                          + NOT_RELAXED + "要不要換個類型，或取消部分條件？", quick_replies=(MORE_BUTTON,))


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
            heads = [_intro(r, _supported(r), criteria) for r in shops]
            intro = lead + "\n" + "\n".join(f"{n}.{text}" for n, (text, _) in enumerate(heads, 1)) + "。"
            blocks = []
            for r in shops:
                lines = [r.name, f"地址：{r.address}", _price(r)]
                if r.map_search_url:
                    lines.append(r.map_search_url)
                blocks.append("\n".join(lines))
            note = GOOGLE_PRICE_NOTE if any(_google_price(r) for r in shops) else MENU_PRICE_NOTE
            if criteria.meal and any(r.meal_periods is None for r in shops):  # Unknown hours: only a preference.
                meal = {"lunch": "午餐", "dinner": "晚餐"}[criteria.meal]
                note += f"是否有供應{meal}尚未確認，出發前請先查營業時間。"
            else:
                note += "資料未向店家確認，出發前請再確認營業時間。"
            if criteria.dishes and criteria.budget_twd is not None and not all(p for _, p in heads):
                # Some shop has no item prices for the dish: its budget check used the shop's cheapest dish.
                note += f"\n※ 預算是用店內最便宜的主餐判斷，不一定是{'、'.join(criteria.dishes)}的價格。"
            return Recommendation("\n\n".join([intro, *blocks, note]), tuple(picks))
        if evaluation.evidence_state in ("insufficient", "no_results"):
            return _no_shop_reply(criteria, retrieval, evaluation)
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
