"""Shared contracts for the future parser, retriever and evaluator."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SearchCriteria:
    county: str = "雲林縣"
    district: str = "斗六市"
    meal: str | None = None
    budget_twd: int | None = None
    cuisines: tuple[str, ...] = ()
    excluded_cuisines: tuple[str, ...] = ()
    landmark: str | None = None


@dataclass(frozen=True)
class Restaurant:
    id: str
    name: str
    county: str
    district: str
    address: str
    source_urls: tuple[str, ...]
    checked_at: str | None
    cuisines: tuple[str, ...] = ()
    price_min_twd: int | None = None
    price_max_twd: int | None = None
    price_evidence: str | None = None
    menu_summary: str = ""
    opening_hours: str | None = None
    landmarks: tuple[str, ...] = ()
    sources: tuple[dict, ...] = ()
    notes: tuple[str, ...] = ()
    map_search_url: str = ""
    meal_periods: tuple[str, ...] | None = None
    operating_status: str = "unknown"
    review_status: str = "documented"


@dataclass(frozen=True)
class Candidate:
    restaurant: Restaurant
    score: float
    evidence: tuple[str, ...]
    matched_terms: tuple[str, ...] = ()
    query_coverage: float = 0.0
    # score is the final ranking score. Without embeddings it equals lexical_score
    # (TF-IDF cosine); with embeddings it is the RRF fusion of both rankings.
    lexical_score: float = 0.0
    semantic_score: float | None = None


@dataclass(frozen=True)
class FilterDecision:
    restaurant_id: str
    outcome: Literal["eligible", "excluded", "unknown", "no_lexical_match"]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    status: Literal["candidates_found", "insufficient_data", "no_results", "empty_dataset"]
    candidates: tuple[Candidate, ...] = ()
    decisions: tuple[FilterDecision, ...] = ()
    total: int = 0
    eligible_count: int = 0
    # "lexical": TF-IDF only; "hybrid_rrf": TF-IDF + embedding fused by RRF;
    # "lexical_fallback": embedding configured but the query embedding failed.
    ranking: Literal["lexical", "hybrid_rrf", "lexical_fallback"] = "lexical"


@dataclass(frozen=True)
class Evaluation:
    decision: Literal["accept", "retry", "clarify", "stop"]
    reasons: tuple[str, ...]
    evidence_state: Literal["sufficient", "insufficient", "no_results", "error"] = "insufficient"
    accepted_ids: tuple[str, ...] = ()
    rewrite_query: str = ""
    model_called: bool = False


@dataclass(frozen=True)
class EvaluationAttempt:
    attempt: int
    query: str
    criteria: SearchCriteria
    retrieval: RetrievalResult
    evaluation: Evaluation


@dataclass(frozen=True)
class ChatResult:
    status: str
    reply: str
    restaurant_ids: tuple[str, ...] = ()
    criteria: SearchCriteria | None = None
    questions: tuple[str, ...] = ()
    retrieval: RetrievalResult | None = None
    evaluation: Evaluation | None = None
    attempts: tuple[EvaluationAttempt, ...] = ()
