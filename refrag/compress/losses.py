"""Loss functions."""
import torch
import torch.nn.functional as F
from typing import Optional

def reconstruction_loss(compressed_vecs: torch.Tensor, target_embeddings: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.mse_loss(compressed_vecs, target_embeddings, reduction='none')
    if mask is not None:
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
    else:
        loss = loss.mean()
    return loss

def contrastive_loss(query_embeds: torch.Tensor, doc_embeds: torch.Tensor,
                    temperature: float = 0.07) -> torch.Tensor:
    q = F.normalize(query_embeds, dim=-1)
    d = F.normalize(doc_embeds, dim=-1)
    logits = torch.matmul(q, d.T) / temperature
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)