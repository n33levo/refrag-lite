"""Chunk encoder built on top of SentenceTransformer."""
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ChunkEncoder(nn.Module):
    """Light wrapper that provides a uniform API around SentenceTransformer.

    The original training code expected an ``encode`` method that returned
    tensors on the target device.  It previously relied on random targets,
    which meant the mismatch went unnoticed.  For the real training pipeline
    we expose both ``forward`` and ``encode`` and make sure gradients stay
    disabled so the sentence-transformer checkpoint is used strictly as a
    frozen teacher model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode ``texts`` into dense embeddings on the target device."""
        if not isinstance(texts, Iterable):
            raise TypeError("texts must be an iterable of strings")
        if len(texts) == 0:
            return torch.empty(0, self.get_embedding_dim(), device=self.device)

        embeddings = self.model.encode(
            list(texts),
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings.to(self.device)

    @torch.no_grad()
    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        """Alias for :meth:`encode` to keep ``nn.Module`` semantics."""
        return self.encode(texts)

    def get_embedding_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())
