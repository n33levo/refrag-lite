"""Chunk encoder."""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List

class ChunkEncoder(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda"):
        super().__init__()
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def get_embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()