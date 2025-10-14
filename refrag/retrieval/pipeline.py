"""Retrieval pipeline."""
from typing import List, Dict, Any, Optional
from refrag.retrieval.hybrid import HybridRetriever

class RetrievalPipeline:
    def __init__(self, dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 bm25_weight: float = 0.5, dense_weight: float = 0.5, device: str = "cuda"):
        self.retriever = HybridRetriever(dense_model, bm25_weight, dense_weight, device)
        self.metadata = []

    def index_corpus(self, corpus: List[str], metadata: Optional[List[Dict]] = None) -> None:
        self.retriever.index(corpus)
        self.metadata = metadata or [{}] * len(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        results = self.retriever.retrieve(query, top_k)
        return [{"idx": idx, "score": score, "text": text,
                 "metadata": self.metadata[idx] if idx < len(self.metadata) else {}}
                for idx, score, text in results]