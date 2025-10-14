"""Adapters for compressed embeddings."""
import torch
import torch.nn as nn

class CompressedEmbeddingAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_tokens = model.get_input_embeddings()

    def forward_with_compressed(self, input_ids: torch.Tensor, compressed_vecs: torch.Tensor,
                               attention_mask: torch.Tensor, compressed_positions: list):
        embeds = self.embed_tokens(input_ids)
        for i, pos in enumerate(compressed_positions):
            if pos < embeds.shape[1]:
                embeds[:, pos, :] = compressed_vecs[:, i, :]
        return self.model(inputs_embeds=embeds, attention_mask=attention_mask)