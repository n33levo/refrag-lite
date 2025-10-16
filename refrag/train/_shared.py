"""Shared helpers for training pipelines."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from refrag.compress.encoder import ChunkEncoder
from refrag.data.hotpotqa import HotpotQADataset


def batch_to_samples(batch: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Convert a dataloader batch into per-sample dictionaries."""
    num_items = len(batch["questions"])
    samples: List[Dict[str, Any]] = []
    for idx in range(num_items):
        samples.append(
            {
                "question": batch["questions"][idx],
                "answer": batch["answers"][idx],
                "context": batch["contexts"][idx],
                "supporting_facts": batch["supporting_facts"][idx],
                "id": batch["ids"][idx],
            }
        )
    return samples


@torch.no_grad()
def prepare_llm_chunk_embeddings(
    llm,
    tokenizer,
    chunks: Sequence[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    """Encode chunks with the LLM and mean-pool the final hidden states."""
    if not chunks:
        return torch.empty(0, llm.config.hidden_size, device=device)

    was_training = llm.training
    if was_training:
        llm.eval()
    tokenized = tokenizer(
        list(chunks),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    outputs = llm(
        **tokenized,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1]
    mask = tokenized["attention_mask"].unsqueeze(-1)
    masked = hidden * mask
    pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    if was_training:
        llm.train()
    return pooled


def collate_compression_targets(
    encoder: ChunkEncoder,
    llm,
    tokenizer,
    samples: Iterable[Dict[str, Any]],
    device: torch.device,
    chunk_char_limit: int,
    context_max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded tensors for compressed vectors and LLM targets."""
    compressed_vectors: List[torch.Tensor] = []
    target_vectors: List[torch.Tensor] = []

    for sample in samples:
        chunks = HotpotQADataset.get_context_chunks(sample, chunk_size=chunk_char_limit)
        if not chunks:
            chunks = [f"Question: {sample['question']}"]

        compressed = encoder.encode(chunks).to(device)
        targets = prepare_llm_chunk_embeddings(
            llm, tokenizer, chunks, device, max_length=context_max_tokens
        )

        if compressed.size(0) == 0 or targets.size(0) == 0:
            continue

        if compressed.size(0) != targets.size(0):
            min_len = min(compressed.size(0), targets.size(0))
            compressed = compressed[:min_len]
            targets = targets[:min_len]

        compressed_vectors.append(compressed)
        target_vectors.append(targets)

    if not compressed_vectors:
        empty_enc = torch.empty(0, encoder.get_embedding_dim(), device=device)
        empty_llm = torch.empty(0, llm.config.hidden_size, device=device)
        empty_mask = torch.empty(0, device=device)
        return empty_enc, empty_llm, empty_mask

    max_chunks = max(vec.size(0) for vec in compressed_vectors)
    encoder_dim = compressed_vectors[0].size(-1)
    llm_dim = target_vectors[0].size(-1)

    padded_compressed: List[torch.Tensor] = []
    padded_targets: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for compressed, targets in zip(compressed_vectors, target_vectors):
        original_len = compressed.size(0)
        pad = max_chunks - original_len
        if pad > 0:
            compressed = torch.cat(
                [compressed, torch.zeros(pad, encoder_dim, device=device)],
                dim=0,
            )
            targets = torch.cat(
                [targets, torch.zeros(pad, llm_dim, device=device)],
                dim=0,
            )
        padded_compressed.append(compressed)
        padded_targets.append(targets)
        mask = torch.zeros(max_chunks, device=device)
        mask[:original_len] = 1.0
        masks.append(mask)

    return (
        torch.stack(padded_compressed),
        torch.stack(padded_targets),
        torch.stack(masks),
    )
