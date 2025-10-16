"""Training script for Stage 3: Supervised fine-tuning on QA pairs."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import tyro
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from refrag.compress.encoder import ChunkEncoder
from refrag.compress.losses import reconstruction_loss
from refrag.compress.projector import Projector
from refrag.data.hotpotqa import HotpotQADataset, create_dataloader
from refrag.llm.model import load_llm
from refrag.train._shared import (
    batch_to_samples,
    collate_compression_targets,
)
from refrag.utils.io import (
    get_device,
    load_yaml as load_config,
    save_checkpoint,
    set_seed,
)


@dataclass
class Args:
    config: str = "configs/default.yaml"


def _select_contexts(
    sample,
    num_expanded: int,
    num_compressed: int,
    chunk_char_limit: int,
) -> Tuple[List[str], List[str]]:
    """Return expanded and compressed context lists for a sample."""
    chunks = HotpotQADataset.get_context_chunks(sample, chunk_size=chunk_char_limit)
    if not chunks:
        surrogate = f"Question: {sample['question']}"
        return [surrogate], []

    expanded = chunks[:num_expanded]
    compressed = chunks[num_expanded : num_expanded + num_compressed]
    return expanded, compressed


def _format_prompt(
    sample,
    expanded_chunks: List[str],
    compressed_chunks: List[str],
) -> str:
    """Build the supervised instruction prompt."""
    lines: List[str] = [
        "You are an expert fact-finding assistant.",
        "Answer the question using the provided evidence.",
        f"Question: {sample['question']}",
    ]

    if expanded_chunks:
        lines.append("Full Evidence:")
        for idx, chunk in enumerate(expanded_chunks, start=1):
            lines.append(f"[Expanded {idx}] {chunk}")

    if compressed_chunks:
        lines.append("Compressed Evidence Embeddings:")
        for idx, chunk in enumerate(compressed_chunks, start=1):
            preview = " ".join(chunk.split()[:40]) + ("..." if len(chunk.split()) > 40 else "")
            lines.append(f"[Compressed {idx}] {preview}")

    lines.append("Provide a concise answer in one sentence.")
    lines.append("Answer:")
    return "\n".join(lines)


def train_sft(config_path: str) -> None:
    """Run Stage 3 supervised fine-tuning."""
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    device = get_device(config.get("device", "auto"))

    print(f"[Stage 3] Loading models on {device}...")
    llm, tokenizer = load_llm(
        config["llm"]["model_name"],
        use_peft=config["llm"].get("use_peft", False),
        peft_config=config["llm"].get("peft_config", {}),
        device=str(device),
        torch_dtype=config["llm"].get("torch_dtype", "float16"),
    )
    llm.train()

    encoder = ChunkEncoder(
        config["encoder"]["model_name"],
        device=str(device),
        normalize=config["encoder"].get("normalize", True),
    )
    encoder_dim = encoder.get_embedding_dim()
    llm_dim = llm.config.hidden_size

    projector = Projector(
        input_dim=encoder_dim,
        output_dim=llm_dim,
        hidden_dims=config["projector"]["hidden_dims"],
        activation=config["projector"]["activation"],
        dropout=config["projector"]["dropout"],
        layer_norm=config["projector"]["layer_norm"],
    ).to(device)

    # Warm-start from continual pretraining if available
    cpt_ckpt = Path(config["checkpoint_dir"]) / "pretrain_cpt.pt"
    if cpt_ckpt.exists():
        try:
            state = torch.load(cpt_ckpt, map_location=device, weights_only=True)
            projector.load_state_dict(state["projector_state_dict"])
            llm.load_state_dict(state["llm_state_dict"], strict=False)
            print(f"[Stage 3] Loaded weights from {cpt_ckpt}")
        except Exception as exc:
            print(f"[Stage 3] Warning: failed to warm-start from Stage 2 ({exc})")

    optimizer = torch.optim.AdamW(
        list(llm.parameters()) + list(projector.parameters()),
        lr=config["sft"]["learning_rate"],
        weight_decay=config["sft"]["weight_decay"],
    )

    grad_accum = max(1, config["sft"].get("gradient_accumulation_steps", 1))
    chunk_char_limit = config["retrieval"].get("chunk_size", 512)
    context_max_tokens = config["encoder"].get("max_length", 256)
    max_seq_len = config["data"].get("max_length", config["llm"].get("max_length", 2048))
    projector_weight = config["sft"].get("projector_weight", 0.15)
    compressed_reg_weight = config["sft"].get("compressed_reg_weight", 0.01)

    num_expanded = max(1, config["sft"].get("num_expanded", 3))
    num_compressed = max(0, config["sft"].get("num_compressed", 7))

    train_samples = config["data"].get("train_samples", None)
    if isinstance(train_samples, int) and train_samples <= 0:
        train_samples = None

    dataloader = create_dataloader(
        config["data"]["path"],
        split="train",
        batch_size=config["sft"]["batch_size"],
        max_samples=train_samples,
        num_workers=config["data"].get("num_workers", 0),
    )

    num_epochs = config["sft"]["num_epochs"]

    print(f"[Stage 3] Supervised fine-tuning for {num_epochs} epochs...")

    cumulative_loss = 0.0
    total_updates = 0
    micro_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        effective_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            samples = batch_to_samples(batch)

            prompts: List[str] = []
            combined_inputs: List[str] = []

            for sample in samples:
                expanded, compressed = _select_contexts(
                    sample,
                    num_expanded=num_expanded,
                    num_compressed=num_compressed,
                    chunk_char_limit=chunk_char_limit,
                )
                prompt = _format_prompt(sample, expanded, compressed)
                answer = str(sample["answer"]).strip()
                combined = f"{prompt} {answer}"
                prompts.append(prompt)
                combined_inputs.append(combined)

            if not combined_inputs:
                continue

            tokenized_full = tokenizer(
                combined_inputs,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(device)

            tokenized_prompts = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)

            prompt_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()
            labels = tokenized_full["input_ids"].clone()
            for idx, prompt_len in enumerate(prompt_lengths):
                labels[idx, :prompt_len] = -100
            labels[tokenized_full["attention_mask"] == 0] = -100

            outputs = llm(
                input_ids=tokenized_full["input_ids"],
                attention_mask=tokenized_full["attention_mask"],
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss

            compressed, targets, mask = collate_compression_targets(
                encoder=encoder,
                llm=llm,
                tokenizer=tokenizer,
                samples=samples,
                device=device,
                chunk_char_limit=chunk_char_limit,
                context_max_tokens=context_max_tokens,
            )

            if compressed.numel() > 0:
                logits = projector(compressed.view(-1, encoder_dim))
                recon_loss = reconstruction_loss(
                    logits,
                    targets.view(-1, llm_dim),
                    mask=mask.view(-1),
                )
                loss = loss + projector_weight * recon_loss

                reg_loss = torch.mean(torch.norm(logits.view(-1, llm_dim), dim=-1))
                loss = loss + compressed_reg_weight * reg_loss

            loss = loss / grad_accum
            loss.backward()

            cumulative_loss += loss.item() * grad_accum
            epoch_loss += loss.item() * grad_accum
            micro_step += 1

            if micro_step % grad_accum == 0:
                clip_grad_norm_(
                    list(llm.parameters()) + list(projector.parameters()),
                    config["sft"]["max_grad_norm"],
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_updates += 1

            effective_batches += 1

        avg_epoch_loss = epoch_loss / max(effective_batches, 1)
        print(f"[Stage 3] Epoch {epoch+1} | avg loss: {avg_epoch_loss:.4f}")

    avg_loss = cumulative_loss / max(total_updates, 1)
    print(f"[Stage 3] SFT finished. Average update loss: {avg_loss:.4f}")

    checkpoint = {
        "llm_state_dict": llm.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "epoch": num_epochs,
        "loss": avg_loss,
    }
    save_checkpoint(checkpoint, f"{config['checkpoint_dir']}/sft_qa.pt")
    print(f"[Stage 3] Saved checkpoint to {config['checkpoint_dir']}/sft_qa.pt")


def main() -> None:
    args = tyro.cli(Args)
    train_sft(args.config)


if __name__ == "__main__":
    main()
