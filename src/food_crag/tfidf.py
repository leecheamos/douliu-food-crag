"""Ported from mini_crag_tutorial/retriever.py and tokenizer.py.

Keep their TF, smoothed IDF, cosine and token contribution formulas. Remove
course-specific paths/normalizations; guard empty input and make ties stable.
"""
import re
import unicodedata
from collections import Counter
from math import log, sqrt

from .criteria import ALIASES, alternatives

ALIAS_PATTERN = re.compile(alternatives(ALIASES))


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = ALIAS_PATTERN.sub(lambda m: ALIASES[m[0]], text)
    tokens = re.findall(r"[a-z0-9]+", text)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend([sequence] if len(sequence) == 1 else
                      [sequence[i:i + 2] for i in range(len(sequence) - 1)])
    return tokens


def calculate_tf(tokens):
    total = len(tokens)
    return {token: count / total for token, count in Counter(tokens).items()} if total else {}


def calculate_idf(tokenized_documents):
    frequencies = Counter()
    for tokens in tokenized_documents:
        frequencies.update(set(tokens))
    count = len(tokenized_documents)
    return {token: log((count + 1) / (frequency + 1)) + 1 for token, frequency in frequencies.items()}


def calculate_tfidf(tokens, idf):
    return {token: frequency * idf[token] for token, frequency in calculate_tf(tokens).items() if token in idf}


def vectorize_query(query, idf):
    return calculate_tfidf(tokenize(query), idf)


def length(vector):
    return sqrt(sum(vector[token] ** 2 for token in sorted(vector)))


def cosine_similarity(vector_a, vector_b):
    denominator = length(vector_a) * length(vector_b)
    if not denominator:
        return 0.0
    return sum(vector_a[t] * vector_b[t] for t in sorted(vector_a.keys() & vector_b.keys())) / denominator


def explain_match(query_vector, document_vector):
    denominator = length(query_vector) * length(document_vector)
    if not denominator:
        return []
    contributions = [(token, query_vector[token] * document_vector[token] / denominator)
                     for token in query_vector.keys() & document_vector.keys()]
    return sorted(contributions, key=lambda pair: (-pair[1], pair[0]))


def calculate_query_coverage(query_vector, document_vector):
    return len(query_vector.keys() & document_vector.keys()) / len(query_vector) if query_vector else 0.0
