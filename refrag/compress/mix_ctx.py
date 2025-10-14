"""Mixed context builder."""
import torch
from typing import Optional
from dataclasses import dataclass

@dataclass
class MixedContextBatch:
    input_embeds: torch.Tensor
    attention_mask: torch.Tensor
    compressed_mask: torch.Tensor

def build_mixed_context(query_ids: torch.Tensor, compressed_vecs: Optional[torch.Tensor],
                       expanded_ids: Optional[torch.Tensor], embedding_layer: torch.nn.Embedding,
                       device: str = "cuda") -> MixedContextBatch:
    batch_size = query_ids.shape[0]
    query_embeds = embedding_layer(query_ids)

    parts = [query_embeds]
    masks = [torch.ones(batch_size, query_ids.shape[1], device=device)]

    if compressed_vecs is not None and compressed_vecs.shape[1] > 0:
        parts.append(compressed_vecs)
        masks.append(torch.ones(batch_size, compressed_vecs.shape[1], device=device))

    if expanded_ids is not None and expanded_ids.shape[1] > 0:
        parts.append(embedding_layer(expanded_ids))
        masks.append(torch.ones(batch_size, expanded_ids.shape[1], device=device))

    input_embeds = torch.cat(parts, dim=1)
    attention_mask = torch.cat(masks, dim=1)

    compressed_mask = torch.zeros_like(attention_mask)
    if compressed_vecs is not None and compressed_vecs.shape[1] > 0:
        start = query_ids.shape[1]
        compressed_mask[:, start:start + compressed_vecs.shape[1]] = 1

    return MixedContextBatch(input_embeds, attention_mask, compressed_mask)