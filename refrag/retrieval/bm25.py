"""BM25 retrieval."""
from typing import List, Tuple
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.corpus = []

    def index(self, corpus: List[str]) -> None:
        self.corpus = corpus
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in top]