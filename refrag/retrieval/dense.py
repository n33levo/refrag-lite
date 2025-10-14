"""Dense retrieval."""
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.corpus = []

    def index_documents(self, corpus: List[str]) -> None:
        import faiss
        self.corpus = corpus
        embeddings = self.model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.index:
            return []
        qemb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(qemb.astype(np.float32), top_k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0])]