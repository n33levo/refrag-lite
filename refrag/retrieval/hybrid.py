"""Hybrid retrieval."""
from typing import List, Tuple
from refrag.retrieval.bm25 import BM25Retriever
from refrag.retrieval.dense import DenseRetriever

class HybridRetriever:
    def __init__(self, dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 bm25_weight: float = 0.5, dense_weight: float = 0.5, device: str = "cuda"):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(dense_model, device)
        self.bm25_w, self.dense_w = bm25_weight, dense_weight
        self.corpus = []

    def index(self, corpus: List[str]) -> None:
        self.corpus = corpus
        self.bm25.index(corpus)
        self.dense.index_documents(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        bm25_res = dict(self.bm25.retrieve(query, top_k * 2))
        dense_res = dict(self.dense.retrieve(query, top_k * 2))

        if bm25_res:
            mx = max(bm25_res.values())
            if mx > 0:
                bm25_res = {k: v/mx for k, v in bm25_res.items()}

        all_idx = set(bm25_res.keys()) | set(dense_res.keys())
        combined = [(i, self.bm25_w * bm25_res.get(i, 0) + self.dense_w * dense_res.get(i, 0),
                    self.corpus[i]) for i in all_idx]
        return sorted(combined, key=lambda x: x[1], reverse=True)[:top_k]