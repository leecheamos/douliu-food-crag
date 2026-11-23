"""Optional dense-embedding ranking signal fused with TF-IDF.

Restaurant vectors are built once offline and cached on disk; at chat time only
the (canonical, repeated) ranking query is embedded and cached in memory.
Embeddings only re-rank candidates that already passed hard filters and have a
lexical hit, so they never add, remove or relax any condition.

Build the cache explicitly (paid API call):
    python -m food_crag.embedding --env-file .env
"""
import argparse
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
from threading import RLock

from .adapters.gemini import GeminiEmbedder, GeminiError
from .config import Settings
from .dataset import load_restaurants
from .retriever import document_text

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "embeddings"
RRF_K = 60  # Standard Reciprocal Rank Fusion constant (Cormack et al., 2009).
MAX_QUERY_CACHE = 256


class EmbeddingCacheError(ValueError):
    pass


def cache_path(model: str, with_google: bool = False) -> Path:
    # Vectors of Google Places shops go to a separate, gitignored cache.
    return CACHE_DIR / (f"{model}.google.json" if with_google else f"{model}.json")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dense_cosine(a, b) -> float:
    if len(a) != len(b):
        raise ValueError("vector dimensions differ")
    denominator = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


def rrf_scores(rankings):
    """Fuse several rankings (each a list of IDs, best first) into one score per ID."""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (RRF_K + rank)
    return scores


class SemanticIndex:
    def __init__(self, model: str, vectors: dict[str, list[float]], embedder):
        self.model = model
        self.vectors = vectors
        self.embedder = embedder
        self._queries: dict[str, list[float]] = {}
        self._lock = RLock()

    def query_vector(self, query: str) -> list[float]:
        """Raises GeminiError on API failure; the retriever then falls back to lexical."""
        with self._lock:
            cached = self._queries.get(query)
        if cached is not None:
            return cached
        vector = self.embedder.embed([query], task_type="RETRIEVAL_QUERY")[0]
        with self._lock:
            if len(self._queries) >= MAX_QUERY_CACHE:
                self._queries.pop(next(iter(self._queries)))
            self._queries[query] = vector
        return vector

    def similarities(self, query: str, restaurant_ids) -> dict[str, float]:
        vector = self.query_vector(query)
        return {rid: dense_cosine(vector, self.vectors[rid]) for rid in restaurant_ids}


def build_cache(restaurants, embedder, model: str, path: Path | None = None) -> Path:
    path = path or cache_path(model)
    texts = [document_text(r) for r in restaurants]
    vectors = embedder.embed(texts, task_type="RETRIEVAL_DOCUMENT")
    payload = {
        "model": model,
        "task_type": "RETRIEVAL_DOCUMENT",
        "documents": {r.id: {"sha256": text_hash(t), "vector": v}
                      for r, t, v in zip(restaurants, texts, vectors, strict=True)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_index(restaurants, embedder, model: str, path: Path | None = None) -> SemanticIndex:
    """Strict: every restaurant must have a vector built from its current text."""
    path = path or cache_path(model)
    if not path.is_file():
        raise EmbeddingCacheError("cache_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = payload["documents"]
        if payload["model"] != model:
            raise EmbeddingCacheError("cache_model_mismatch")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, EmbeddingCacheError):
            raise
        raise EmbeddingCacheError("cache_invalid") from None
    vectors, dimensions = {}, None
    for restaurant in restaurants:
        entry = documents.get(restaurant.id)
        if not isinstance(entry, dict) or entry.get("sha256") != text_hash(document_text(restaurant)):
            raise EmbeddingCacheError("cache_stale")
        vector = entry.get("vector")
        if (not isinstance(vector, list) or not vector or
                not all(isinstance(x, (int, float)) and isfinite(x) for x in vector)):
            raise EmbeddingCacheError("cache_invalid")
        if dimensions is not None and len(vector) != dimensions:
            raise EmbeddingCacheError("cache_invalid")
        dimensions = len(vector)
        vectors[restaurant.id] = [float(x) for x in vector]
    return SemanticIndex(model, vectors, embedder)


def load_from_settings(restaurants, settings: Settings):
    """Returns (index or None, status). Never raises: chat must work without embeddings."""
    if not settings.embedding_model:
        return None, "disabled"
    try:
        path = cache_path(settings.embedding_model, bool(settings.google_data_file))
        return load_index(restaurants, GeminiEmbedder(settings), settings.embedding_model, path), "hybrid_rrf"
    except EmbeddingCacheError as error:
        return None, str(error)


def main():
    parser = argparse.ArgumentParser(description="Build the restaurant embedding cache (paid API call).")
    parser.add_argument("--env-file")
    args = parser.parse_args()
    settings = Settings.from_env(args.env_file)
    if not settings.embedding_model:
        print("EMBEDDING_MODEL is empty; set it (e.g. gemini-embedding-001) first.")
        return 1
    restaurants = load_restaurants(extra=settings.google_data_file or None)
    try:
        path = build_cache(restaurants, GeminiEmbedder(settings), settings.embedding_model,
                           cache_path(settings.embedding_model, bool(settings.google_data_file)))
    except GeminiError as error:
        print(f"Embedding failed: {error}")
        return 1
    print(f"Wrote {len(restaurants)} vectors to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
