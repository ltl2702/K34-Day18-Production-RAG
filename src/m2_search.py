from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


class _FallbackBM25:
    """Small lexical fallback used only when rank-bm25 is unavailable."""

    def __init__(self, corpus_tokens: list[list[str]]):
        self.corpus_tokens = corpus_tokens

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        query_terms = set(query_tokens)
        return [float(sum(token in query_terms for token in document)) for document in self.corpus_tokens]


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").replace("_", " ")
    except Exception as exc:
        print(f"  ⚠️  Vietnamese segmenter unavailable; using whitespace fallback: {exc}")
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(str(chunk.get("text", ""))).lower().split()
            for chunk in self.documents
        ]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            print("  ⚠️  rank-bm25 is unavailable; using lexical BM25 fallback.")
            self.bm25 = _FallbackBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        query_tokens = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        return [
            SearchResult(
                text=str(self.documents[index].get("text", "")),
                score=float(scores[index]),
                metadata=dict(self.documents[index].get("metadata", {})),
                method="bm25",
            )
            for index in top_indices
            if scores[index] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            texts = [str(chunk.get("text", "")) for chunk in chunks]
            vectors = self._get_encoder().encode(texts, show_progress_bar=False)
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=index,
                    vector=vector.tolist(),
                    payload={**dict(chunk.get("metadata", {})), "text": texts[index]},
                )
                for index, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            self.client.upsert(collection_name=collection, points=points, wait=True)
        except Exception as exc:
            print(f"  ⚠️  Dense indexing unavailable; BM25 remains active: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
            return [
                SearchResult(
                    text=str(point.payload.get("text", "")),
                    score=float(point.score),
                    metadata=dict(point.payload),
                    method="dense",
                )
                for point in response.points
                if point.payload and point.payload.get("text")
            ]
        except Exception as exc:
            print(f"  ⚠️  Dense search unavailable; returning BM25-only hybrid: {exc}")
            return []


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda entry: entry["score"], reverse=True)
    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=dict(entry["result"].metadata),
            method="hybrid",
        )
        for entry in ranked[:max(top_k, 0)]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
