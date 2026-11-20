"""Interfaces only; no claim of implemented retrieval or model calls."""
from typing import Protocol
from .domain import Candidate, Evaluation, RetrievalResult, SearchCriteria
from .criteria import ParseResult


class LanguageModel(Protocol):
    def generate_json(self, payload: dict, schema: dict) -> dict: ...


class Embedder(Protocol):
    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]: ...


class CriteriaParser(Protocol):
    def parse(self, message: str, previous: SearchCriteria | None) -> ParseResult: ...


class Retriever(Protocol):
    def search(self, query: str, criteria: SearchCriteria, limit: int = 10) -> RetrievalResult: ...


class Evaluator(Protocol):
    def evaluate(self, criteria: SearchCriteria, report: RetrievalResult) -> Evaluation: ...


class ReplyGenerator(Protocol):
    """Returns recommendation.Recommendation; output is verified before use."""
    def generate(self, criteria: SearchCriteria, retrieval: RetrievalResult | None,
                 evaluation: Evaluation | None): ...


class ConversationStore(Protocol):
    def get(self, user_id: str) -> SearchCriteria | None: ...
    def set(self, user_id: str, criteria: SearchCriteria) -> None: ...
    def reset(self, user_id: str) -> None: ...
