"""Structured filtering followed by prebuilt TF-IDF ranking; no CRAG evaluator.

With an optional SemanticIndex, candidates (still: eligible AND lexical hit) are
re-ranked by Reciprocal Rank Fusion of the TF-IDF and embedding rankings.
"""
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


def sells(restaurant, dish):
    """Literal containment only (user decision 2026-11-21): 厚切豬排飯 sells 豬排飯; 排骨飯 does not."""
    texts = (restaurant.name, restaurant.menu_summary, *restaurant.menu_items)
    return any(dish in text for text in texts)


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
            # Prices are listed main-dish single-item prices, not per-person
            # meal cost: a shop fits when its cheapest main dish is in budget.
            if not restaurant.price_evidence or "price_evidence" not in supported or low is None:
                unknown.append("缺少有來源的主餐標價資料")
            elif low > criteria.budget_twd:
                excluded.append("最便宜的主餐標價超出預算")
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
