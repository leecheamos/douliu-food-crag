"""Local veto, Gemini evidence assessment and bounded equivalent-query retries."""
from dataclasses import asdict, replace
from math import isclose, isfinite

from .adapters.gemini import GeminiError
from .criteria import ALIASES, PARENTS
from .domain import Evaluation, EvaluationAttempt
from .retriever import RestaurantRetriever
from .tfidf import calculate_query_coverage, cosine_similarity, explain_match, vectorize_query

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "retry", "clarify", "stop"]},
        "evidence_state": {"type": "string", "enum": ["sufficient", "insufficient", "no_results"]},
        "accepted_ids": {"type": "array", "items": {"type": "string"}},
        "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10},
        "rewrite_query": {"type": "string", "description": "Must be empty unless decision is retry; then use one allowed_rewrites value."},
    },
    "required": ["decision", "evidence_state", "accepted_ids", "reasons", "rewrite_query"],
}
CHILDREN = {}
for _child, _parent in PARENTS.items():
    CHILDREN.setdefault(_parent, []).append(_child)


def allowed_rewrites(query, criteria=None):
    """Expand a parent cuisine with its narrower sub-categories to re-rank.

    Synonyms are already normalized by the tokenizer, so swapping them cannot
    change results. Sub-categories are narrower than the request, so adding
    them never relaxes a hard condition; excluded cuisines are never added.
    """
    words = query.split()
    excluded = {ALIASES.get(c, c) for c in (criteria.excluded_cuisines if criteria else ())}
    extra = [child for word in words for child in CHILDREN.get(ALIASES.get(word, word), ())
             if child not in words and child not in excluded]
    return (" ".join((*words, *dict.fromkeys(extra))),) if extra else ()


class GeminiEvaluator:
    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model
        self.catalog = {r.id: r for r in retriever.restaurants}
        self.vectors = {r.id: vector for r, vector in retriever.index}

    def evaluate(self, criteria, report):
        if not report.candidates:
            checks = [RestaurantRetriever.filter(r, criteria) for r in self.catalog.values()]
            if any(check.outcome == "unknown" for check in checks):
                return Evaluation("clarify", ("必要的價格、料理、供餐時段或地標資料不足；需補充資料後才能判斷。",), "insufficient")
            return Evaluation("stop", ("沒有符合硬條件且有詞彙命中的候選，停止檢索。",), "no_results")
        hybrid = report.ranking == "hybrid_rrf"
        allowed = {"hybrid_rrf", "lexical_fallback"} if self.retriever.semantic is not None else {"lexical"}
        if report.ranking not in allowed:
            return Evaluation("stop", ("候選排序模式與本機檢索設定不符。",), "error")
        seen = set()
        query_vector = vectorize_query(report.query, self.retriever.idf)
        for candidate in report.candidates:
            restaurant = self.catalog.get(candidate.restaurant.id)
            if restaurant is None or restaurant != candidate.restaurant or restaurant.id in seen:
                return Evaluation("stop", ("候選 ID、內容或唯一性與本機資料不符。",), "error")
            seen.add(restaurant.id)
            check = RestaurantRetriever.filter(restaurant, criteria)
            if check.outcome != "eligible":
                return Evaluation("stop", ("候選未通過本機硬條件覆核。", *check.reasons), "error")
            supported = {field for s in restaurant.sources for field in s.get("supports", ())}
            score = cosine_similarity(query_vector, self.vectors[restaurant.id])
            contributions = explain_match(query_vector, self.vectors[restaurant.id])
            terms = tuple(token for token, _ in contributions)
            evidence = tuple(f"{token}：相似度貢獻 {value:.6f}" for token, value in contributions)
            if hybrid:
                semantic = candidate.semantic_score
                fusion_ok = (isinstance(semantic, float) and isfinite(semantic) and -1 - 1e-9 <= semantic <= 1 + 1e-9)
            else:
                fusion_ok = (candidate.semantic_score is None and
                             isclose(candidate.score, candidate.lexical_score, rel_tol=1e-9, abs_tol=1e-12))
            if (not isfinite(candidate.score) or candidate.score <= 0 or not fusion_ok or
                not isclose(candidate.lexical_score, score, rel_tol=1e-9, abs_tol=1e-12) or
                candidate.evidence != evidence or candidate.matched_terms != terms or
                not isclose(candidate.query_coverage, calculate_query_coverage(query_vector, self.vectors[restaurant.id])) or
                not {"name", "address", "cuisines"} <= supported):
                return Evaluation("stop", ("候選詞彙分數、命中證據或基本來源不完整。",), "error")
        if hybrid:
            # Fused RRF scores depend on the whole ranking, so recompute it. The
            # query embedding is cached, so this normally makes no API call.
            fresh = self.retriever.search(report.query, criteria, limit=len(report.candidates))
            if fresh.ranking != "hybrid_rrf" or fresh.candidates != report.candidates:
                return Evaluation("stop", ("候選語意分數或融合排序與本機重算結果不符。",), "error")
        rewrites = allowed_rewrites(report.query, criteria)
        limits = ("Source review is not current operation verification; categories are not a complete menu. "
                  "price_min_twd/price_max_twd are listed main-dish single-item prices from menu photos, not "
                  "per-person meal cost; a budget is met when price_min_twd is within it.")
        if hybrid:
            limits += (" semantic_score is embedding similarity used only to re-rank candidates that already "
                       "passed hard filters; it is not evidence that any condition is met.")
        payload = {
            "criteria": asdict(criteria), "query": report.query, "allowed_rewrites": list(rewrites),
            "ranking": report.ranking,
            "candidates": [asdict(candidate) for candidate in report.candidates],
            "limits": limits,
        }
        try:
            output = self.model.generate_json(payload, SCHEMA)
        except GeminiError as error:
            attempted = str(error) not in {"not_configured", "invalid_model_name"}
            return Evaluation("stop", (f"Gemini 評估未完成：{error}。未以本機結果冒充模型通過。",), "error", model_called=attempted)
        if not self.valid_output(output, seen, rewrites):
            return Evaluation("stop", ("Gemini 評估格式、候選 ID 或修正查詢不符合限制。",), "error", model_called=True)
        return Evaluation(output["decision"], tuple(output["reasons"]), output["evidence_state"],
                          tuple(output["accepted_ids"]), output["rewrite_query"], True)

    @staticmethod
    def valid_output(output, candidate_ids, rewrites):
        if not isinstance(output, dict) or set(output) != set(SCHEMA["required"]):
            return False
        decision, state = output["decision"], output["evidence_state"]
        if not isinstance(decision, str) or decision not in {"accept", "retry", "clarify", "stop"}:
            return False
        if not isinstance(state, str) or state not in {"sufficient", "insufficient", "no_results"}:
            return False
        ids, reasons, query = output["accepted_ids"], output["reasons"], output["rewrite_query"]
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            return False
        if len(ids) != len(set(ids)) or not set(ids) <= candidate_ids:
            return False
        if not isinstance(reasons, list) or not 1 <= len(reasons) <= 10 or not all(isinstance(r, str) and 0 < len(r.strip()) <= 500 for r in reasons):
            return False
        if not isinstance(query, str) or len(query) > 160:
            return False
        if decision == "accept":
            return state == "sufficient" and bool(ids) and not query
        if ids or state == "sufficient":
            return False
        if decision == "retry":
            return bool(query) and query in rewrites
        return not query and (decision != "clarify" or state == "insufficient")


def retrieve_and_evaluate(retriever, evaluator, criteria, query):
    attempts = []
    for number in (0, 1):
        report = retriever.search(query, criteria)
        evaluation = evaluator.evaluate(criteria, report)
        if evaluation.decision == "retry" and number == 1:
            evaluation = replace(evaluation, decision="stop", rewrite_query="", accepted_ids=(),
                                 reasons=("已達一次修正上限，停止；未放寬條件。", *evaluation.reasons))
        attempts.append(EvaluationAttempt(number, query, criteria, report, evaluation))
        if evaluation.decision != "retry":
            break
        if evaluation.rewrite_query not in allowed_rewrites(query, criteria) or evaluation.rewrite_query == query:
            evaluation = replace(evaluation, decision="stop", evidence_state="error", rewrite_query="",
                                 reasons=("修正查詢未通過本機白名單。",))
            attempts[-1] = replace(attempts[-1], evaluation=evaluation)
            break
        query = evaluation.rewrite_query
    return report, evaluation, tuple(attempts)
