"""Semantic retrieval with Sentence Transformers and an offline fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .documents import instruction_text
from .text_processing import char_ngrams, normalize_text, tokenize

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class SearchResult:
    item: dict[str, Any] | None
    score: float
    engine: str


class SemanticSearchEngine:
    """Find a controlled answer by meaning; never generates new facts."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        model_name: str = DEFAULT_MODEL,
        min_similarity: float = 0.45,
        use_transformer: bool = True,
    ) -> None:
        self.items = items
        self.model_name = model_name
        self.min_similarity = min_similarity
        self.model = None
        self.embeddings = None
        self.engine_name = "Локальный лексический режим"
        if use_transformer:
            self._load_transformer()

    def _document_text(self, item: dict[str, Any]) -> str:
        aliases = " ".join(item.get("aliases", []))
        keywords = " ".join(item.get("keywords", []))
        return f"{item['question']} {aliases} {keywords} {item['category']}"

    def _load_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            documents = [self._document_text(item) for item in self.items]
            self.embeddings = self.model.encode(documents, normalize_embeddings=True)
            self.engine_name = "Sentence Transformers · MiniLM-L12-v2"
        except (ImportError, OSError, RuntimeError):
            self.model = None
            self.embeddings = None

    def search(self, query: str, category: str | None = None) -> SearchResult:
        query = normalize_text(query)
        if not query:
            return SearchResult(None, 0.0, self.engine_name)
        indexes = [
            index for index, item in enumerate(self.items)
            if not category or category == "Все категории" or item["category"] == category
        ]
        if not indexes:
            return SearchResult(None, 0.0, self.engine_name)

        if self.model is not None and self.embeddings is not None:
            query_vector = self.model.encode([query], normalize_embeddings=True)[0]
            scored = [(index, float(self.embeddings[index] @ query_vector)) for index in indexes]
        else:
            scored = [(index, self._lexical_score(query, self.items[index])) for index in indexes]

        best_index, best_score = max(scored, key=lambda pair: pair[1])
        threshold = self.min_similarity if self.model is not None else 0.18
        item = self.items[best_index] if best_score >= threshold else None
        return SearchResult(item, max(0.0, min(1.0, best_score)), self.engine_name)

    def _lexical_score(self, query: str, item: dict[str, Any]) -> float:
        document = self._document_text(item)
        query_tokens = set(tokenize(query))
        document_tokens = set(tokenize(document))
        token_score = _jaccard(query_tokens, document_tokens)
        ngram_score = _jaccard(char_ngrams(query), char_ngrams(document))
        phrase_bonus = 0.0
        normalized_document = normalize_text(document)
        for keyword in item.get("keywords", []):
            if normalize_text(keyword) in query or query in normalize_text(keyword):
                phrase_bonus += 0.18
        if query in normalized_document:
            phrase_bonus += 0.25
        return min(1.0, token_score * 1.7 + ngram_score * 0.7 + phrase_bonus)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


class DocumentSearchEngine:
    """Evolution of the legacy normalized-vector dot-product search into fragment top-k.

    The original SemanticSearchEngine remains available for regression tests and
    separate MiniLM comparisons. Its threshold/fallback is not used for E5.
    """

    def __init__(self, encoder, chunks=None, embeddings=None):
        self.encoder = encoder
        self.chunks = chunks or []
        self.embeddings = embeddings
        self.engine_name = f'{encoder.signature["model_id"]} · {encoder.device.upper()} · {encoder.precision.upper()}'

    @staticmethod
    def _document_text(item):
        return instruction_text(item)

    def build(self, chunks, batch_size=32):
        # Publish the new state only after successful inference.
        embeddings = self.encoder.encode([c['text'] for c in chunks], batch_size=batch_size)
        self.chunks, self.embeddings = chunks, embeddings
        return embeddings

    def search(self, query, top_k=5):
        import numpy as np

        from .errors import IndexError, NavigatorError
        query = query.strip()
        if not query:
            raise NavigatorError('Введите непустой поисковый запрос.')
        if self.embeddings is None or not self.chunks:
            raise IndexError('Сначала постройте или откройте сохранённый индекс.')
        if not 1 <= top_k <= 5:
            raise NavigatorError('Можно получить от одного до пяти фрагментов.')
        query_vector = self.encoder.encode([query], query=True, batch_size=1)[0]
        scores = self.embeddings @ query_vector
        indexes = np.argsort(-scores, kind='stable')[:top_k]
        return [SearchResult(self.chunks[i], float(scores[i]), self.engine_name) for i in indexes]
