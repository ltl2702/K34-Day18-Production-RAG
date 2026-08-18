from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._load_attempted = False

    def _load_model(self):
        if self._model is None and not self._load_attempted:
            self._load_attempted = True
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception as exc:
                print(f"  ⚠️  CrossEncoder unavailable; using lexical rerank fallback: {exc}")
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        if model is not None:
            scores = model.predict([(query, str(document.get("text", ""))) for document in documents])
            if isinstance(scores, (int, float)):
                scores = [scores]
        else:
            query_terms = set(query.lower().split())
            scores = [
                len(query_terms.intersection(document.get("text", "").lower().split()))
                / max(len(query_terms), 1)
                for document in documents
            ]

        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=str(document.get("text", "")),
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [{"id": index, "text": str(document.get("text", ""))} for index, document in enumerate(documents)]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            return [
                RerankResult(
                    text=result["text"],
                    original_score=float(documents[int(result["id"])].get("score", 0.0)),
                    rerank_score=float(result["score"]),
                    metadata=dict(documents[int(result["id"])].get("metadata", {})),
                    rank=rank,
                )
                for rank, result in enumerate(results[:top_k])
            ]
        except Exception as exc:
            print(f"  ⚠️  Flashrank unavailable; returning no Flashrank candidates: {exc}")
            return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
