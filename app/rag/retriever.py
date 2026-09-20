"""Recuperación híbrida: similitud densa (Pinecone) + re-ranking léxico, fusionados con RRF."""

import re
import unicodedata
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.schemas import Candidate

RRF_K = 60

_STOPWORDS = frozenset(
    "de la el los las y o en con para por un una que del al se es a su sus lo como mas más sin".split()
)


@dataclass
class RetrievedChunk:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    async def query(
        self, vector: list[float], top_k: int, filter: Mapping[str, Any] | None = None
    ) -> list[RetrievedChunk]: ...


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9+#]+", normalized) if t not in _STOPWORDS and len(t) > 1]


def lexical_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(tokenize(text))) / len(query_tokens)


def fuse_rankings(dense: list[RetrievedChunk], query: str) -> list[tuple[RetrievedChunk, float]]:
    """Reciprocal Rank Fusion entre el orden denso (Pinecone) y el orden léxico (solapamiento de términos)."""
    q_tokens = set(tokenize(query))
    lexical_order = sorted(dense, key=lambda c: lexical_score(q_tokens, c.text), reverse=True)
    fused: dict[str, float] = defaultdict(float)
    for ranking in (dense, lexical_order):
        for rank, chunk in enumerate(ranking, start=1):
            fused[chunk.id] += 1.0 / (RRF_K + rank)
    by_id = {c.id: c for c in dense}
    return sorted(((by_id[i], s) for i, s in fused.items()), key=lambda x: x[1], reverse=True)


class CandidateRetriever:
    """Devuelve candidatos únicos (perfil estructurado en metadatos) ordenados por relevancia."""

    def __init__(
        self,
        embed_query: Callable[[str], Awaitable[list[float]]],
        store: VectorStore,
        chunks_per_candidate: int = 4,
    ) -> None:
        self._embed_query = embed_query
        self._store = store
        self._chunks_per_candidate = chunks_per_candidate

    async def search(self, query: str, top_k: int = 5, min_experience_years: int | None = None) -> list[Candidate]:
        vector = await self._embed_query(query)
        metadata_filter = (
            {"experience_years": {"$gte": min_experience_years}} if min_experience_years is not None else None
        )
        chunks = await self._store.query(vector, top_k * self._chunks_per_candidate, metadata_filter)

        best: dict[str, tuple[float, Candidate]] = {}
        for chunk, score in fuse_rankings(chunks, query):
            cid = chunk.metadata.get("candidate_id", chunk.id)
            if cid not in best:
                best[cid] = (score, Candidate.model_validate(chunk.metadata))
        ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
        return [candidate for _, candidate in ranked[:top_k]]
