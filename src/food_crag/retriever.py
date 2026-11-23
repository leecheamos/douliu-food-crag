"""Structured filtering followed by prebuilt TF-IDF ranking; no CRAG evaluator.

With an optional SemanticIndex, candidates (still: eligible AND lexical hit) are
re-ranked by Reciprocal Rank Fusion of the TF-IDF and embedding rankings.
"""
import re
from dataclasses import replace

from .adapters.gemini import GeminiError
from .criteria import ALIASES, PARENTS
from .domain import Candidate, FilterDecision, RetrievalResult, SearchCriteria
from .tfidf import (
    calculate_idf, calculate_query_coverage, calculate_tfidf, cosine_similarity,
    explain_match, tokenize, vectorize_query,
)


def cuisine_labels(cuisines):
    labels = {ALIASES.get(name, name) for name in cuisines}
    return labels | {PARENTS[name] for name in labels if name in PARENTS}


def ranking_query(criteria: SearchCriteria):
    # Reconstruct from accumulated positive conditions, never from a negative
    # instruction or a later "change budget" message. Region-only is browsing.
    return " ".join((*criteria.cuisines, *criteria.dishes,
                     *((criteria.landmark,) if criteria.landmark else ()))) or criteria.district


def document_text(r):
    """The single text representation shared by TF-IDF and embeddings."""
    return " ".join((
        r.name, r.county, r.district, *sorted(cuisine_labels(r.cuisines)),
        r.menu_summary, *r.menu_items, *r.landmarks,
    ))


STAPLES = ("麵線", "便當", "飯", "麵", "丼")


def dish_in(dish, text):
    """True if one menu text is the dish (user decision 2026-11-21).

    A dish ending in a staple with a 2+ character main part also matches when
    that part comes before the item's final staple: 咖哩豬排蛋包飯 counts as 豬排飯, while
    排骨飯 does not (no 豬排), 咖哩炸飯糰 is not 咖哩飯 (ends in 糰) and
    炒飯 stays literal (one-character part).
    """
    if dish in text:
        return True
    staple = next((s for s in STAPLES if dish.endswith(s)), None)
    head = dish[:-len(staple)] if staple else ""
    text = re.sub(r"[（(].*$", "", text)  # 咖哩豬排蛋包飯（大） still ends in its staple.
    start = text.find(head) if len(head) >= 2 else -1
    return start >= 0 and text.endswith(staple) and start + len(head) <= len(text) - len(staple)


def sells(restaurant, dish):
    """Checked per item, so 炒麵、滷肉飯 in one summary never adds up to 炒飯."""
    texts = (restaurant.name, *restaurant.menu_summary.split("、"), *restaurant.menu_items)
    return any(dish_in(dish, text) for text in texts)


NO_PRICE = "缺少有來源的主餐標價資料"
ROUGH_PRICE = "只有 Google 粗略價位，無法確認是否在預算內"
BUDGET_UNKNOWN = {NO_PRICE, ROUGH_PRICE}
OVER_BUDGET = {"最便宜的主餐標價超出預算", "符合的品項標價都超出預算"}

# Words that make a menu item count as the broad 麵食/飯食 types (user decision 2026-11-23).
ITEM_WORDS = {"麵食": ("麵", "米粉", "冬粉", "粄條", "粿仔"), "飯食": ("飯", "粥", "米糕", "丼")}


def fitting_items(restaurant, criteria):
    """(item, price) pairs of the requested kind, in menu order; None when the kind cannot be told from names.

    Dishes match by dish_in; 麵食/飯食 by ITEM_WORDS; a cuisine ending in a staple
    (牛肉麵, 豬腳飯) by dish_in. Other cuisines (火鍋, 日式料理) have no item rule, so
    every priced item counts. No request at all also counts every priced item.
    """
    if criteria.dishes:
        return [(i, p) for i, p in restaurant.menu_prices if any(dish_in(d, i) for d in criteria.dishes)]
    rules = []
    for cuisine in (ALIASES.get(c, c) for c in criteria.cuisines):
        if cuisine in ITEM_WORDS:
            rules.append(lambda i, words=ITEM_WORDS[cuisine]: any(w in i for w in words))
        elif cuisine.endswith(STAPLES):
            rules.append(lambda i, c=cuisine: dish_in(c, i))
        else:
            return None
    return [(i, p) for i, p in restaurant.menu_prices if not rules or any(rule(i) for rule in rules)]


class RestaurantRetriever:
    def __init__(self, restaurants, semantic=None):
        self.restaurants = tuple(restaurants)
        ids = [r.id for r in self.restaurants]
        if len(ids) != len(set(ids)):
            raise ValueError("Restaurant IDs must be unique")
        if semantic is not None and set(semantic.vectors) != set(ids):
            raise ValueError("Semantic index must cover exactly the loaded restaurants")
        self.semantic = semantic
        # Keep each restaurant and its own vector together: no zip truncation.
        documents = [tokenize(document_text(r)) for r in self.restaurants]
        self.idf = calculate_idf(documents)
        self.index = tuple((r, calculate_tfidf(tokens, self.idf))
                           for r, tokens in zip(self.restaurants, documents, strict=True))

    @staticmethod
    def filter(restaurant, criteria):
        excluded, unknown = [], []
        if (restaurant.county, restaurant.district) != (criteria.county, criteria.district):
            excluded.append("地區不符")
        labels = cuisine_labels(restaurant.cuisines)
        # A shop that literally sells a requested dish satisfies the cuisine condition.
        if criteria.cuisines and not any(sells(restaurant, d) for d in criteria.dishes):
            requested = {ALIASES.get(c, c) for c in criteria.cuisines}
            if not requested & labels:
                if not labels or any(PARENTS.get(c) in labels for c in requested):
                    unknown.append("料理細分類資料不足")
                else:
                    excluded.append("料理分類不符")
        for name in criteria.excluded_cuisines:
            name = ALIASES.get(name, name)
            if name in labels:
                excluded.append(f"命中排除料理：{name}")
            elif not labels or PARENTS.get(name) in labels:
                unknown.append(f"無法確認是否含排除料理：{name}")
        supported = {field for source in restaurant.sources for field in source.get("supports", ())}
        if criteria.budget_twd is not None:
            low, high = restaurant.price_min_twd, restaurant.price_max_twd
            # Prices are listed single-item prices, not per-person meal cost. With item
            # prices the requested kind must fit; otherwise the cheapest main dish decides.
            fitting = fitting_items(restaurant, criteria) if "price_evidence" in supported else None
            if fitting:  # Item prices known: the requested kind itself must fit the budget.
                if min(p for _, p in fitting) > criteria.budget_twd:
                    excluded.append("符合的品項標價都超出預算")
            elif not restaurant.price_evidence or "price_evidence" not in supported or low is None:
                unknown.append(NO_PRICE)
            elif low > criteria.budget_twd:
                excluded.append("最便宜的主餐標價超出預算")
            elif low <= 1 and criteria.budget_twd < high:  # Google's "up to N" bracket says nothing below N.
                unknown.append(ROUGH_PRICE)
        # Meal is a preference (user decision 2026-11-20): almost no shop has
        # sourced service periods, so unknown periods pass and the reply asks the
        # user to check opening hours. Only a sourced mismatch excludes a shop.
        if criteria.meal and restaurant.meal_periods is not None and "meal_periods" in supported                 and criteria.meal not in restaurant.meal_periods:
            excluded.append("用餐時段不符")
        if criteria.landmark:
            if not restaurant.landmarks or "landmarks" not in supported:
                unknown.append("地標資料未知")
            elif criteria.landmark not in restaurant.landmarks:
                excluded.append("地標標記不符")
        return FilterDecision(restaurant.id, "excluded" if excluded else "unknown" if unknown else "eligible",
                              tuple(excluded + unknown))

    def search(self, query, criteria, limit=10):
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(query, str):
            raise ValueError("query must be text")
        if criteria.budget_twd is not None and (type(criteria.budget_twd) is not int or criteria.budget_twd <= 0):
            raise ValueError("budget must be a positive integer or None")
        if criteria.meal not in {None, "lunch", "dinner"}:
            raise ValueError("unsupported meal period")
        query_vector = vectorize_query(query, self.idf)
        candidates, decisions = [], []
        eligible_count = 0
        for restaurant, vector in self.index:
            decision = self.filter(restaurant, criteria)
            if decision.outcome != "eligible":
                decisions.append(decision)
                continue
            eligible_count += 1
            score = cosine_similarity(query_vector, vector)
            if score <= 0:
                decisions.append(FilterDecision(restaurant.id, "no_lexical_match", ("查詢與店家資料無詞彙命中",)))
                continue
            contributions = explain_match(query_vector, vector)
            candidates.append(Candidate(
                restaurant, score,
                tuple(f"{token}：相似度貢獻 {value:.6f}" for token, value in contributions),
                tuple(token for token, _ in contributions),
                calculate_query_coverage(query_vector, vector),
                lexical_score=score,
            ))
            decisions.append(decision)
        candidates.sort(key=lambda c: (-c.score, c.restaurant.id))
        candidates, ranking = self._fuse(query, candidates)
        status = ("empty_dataset" if not self.restaurants else
                  "candidates_found" if candidates else
                  "insufficient_data" if any(d.outcome == "unknown" for d in decisions) else "no_results")
        return RetrievalResult(query, status, tuple(candidates[:limit]), tuple(decisions),
                               len(self.restaurants), eligible_count, ranking)

    def _fuse(self, query, lexical_ranked):
        if self.semantic is None:
            return lexical_ranked, "lexical"
        if not lexical_ranked:
            return lexical_ranked, "hybrid_rrf"
        try:
            semantic = self.semantic.similarities(query, [c.restaurant.id for c in lexical_ranked])
        except GeminiError:
            return lexical_ranked, "lexical_fallback"
        semantic_ranked = sorted(lexical_ranked, key=lambda c: (-semantic[c.restaurant.id], c.restaurant.id))
        from .embedding import rrf_scores  # Local import: embedding imports this module.
        fused = rrf_scores([[c.restaurant.id for c in lexical_ranked],
                            [c.restaurant.id for c in semantic_ranked]])
        result = [replace(c, score=fused[c.restaurant.id], semantic_score=semantic[c.restaurant.id])
                  for c in lexical_ranked]
        # Ties fall back to the lexical score, then ID, so ordering stays deterministic.
        result.sort(key=lambda c: (-c.score, -c.lexical_score, c.restaurant.id))
        return result, "hybrid_rrf"
