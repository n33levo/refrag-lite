"""Training script for Stage 2: Continual Pretraining with mixed contexts."""
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


def _split_context_chunks(
    sample,
    num_expanded: int,
    chunk_char_limit: int,
) -> Tuple[List[str], List[str]]:
    """Split sample contexts into expanded and compressed subsets."""
    all_chunks = HotpotQADataset.get_context_chunks(sample, chunk_size=chunk_char_limit)
    if not all_chunks:
        surrogate = f"Question: {sample['question']}"
        return [surrogate], []

    expanded = all_chunks[:num_expanded]
    compressed = all_chunks[num_expanded:]
    return expanded, compressed


def _summarize_chunk(text: str, max_words: int = 40) -> str:
    """Return a lightweight summary used for compressed context hints."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    summary = " ".join(words[:max_words]) + "..."
    return summary.strip()


def _build_prompt(sample, expanded_chunks: List[str], compressed_chunks: List[str]) -> str:
    """Construct the textual prompt used for continual pretraining."""
    lines: List[str] = [f"Question: {sample['question']}"]

    if expanded_chunks:
        for idx, chunk in enumerate(expanded_chunks, start=1):
            lines.append(f"Expanded Context {idx}: {chunk}")

    if compressed_chunks:
        lines.append("Compressed Context Summaries:")
        for idx, chunk in enumerate(compressed_chunks, start=1):
            lines.append(f"- Summary {idx}: {_summarize_chunk(chunk)}")

    lines.append("Answer:")
    return "\n".join(lines)


def train_cpt(config_path: str) -> None:
    """Train with continual pretraining using curriculum over context expansion."""
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    device = get_device(config.get("device", "auto"))

    print(f"[Stage 2] Loading models on {device}...")
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

    # Warm-start projector from Stage 1 if available
    recon_ckpt = Path(config["checkpoint_dir"]) / "pretrain_recon.pt"
    if recon_ckpt.exists():
        try:
            state = torch.load(recon_ckpt, map_location=device, weights_only=True)
            projector.load_state_dict(state["projector_state_dict"])
            print(f"[Stage 2] Loaded projector initialization from {recon_ckpt}")
        except Exception as exc:
            print(f"[Stage 2] Warning: failed to load Stage 1 checkpoint ({exc})")

    optimizer = torch.optim.AdamW(
        list(llm.parameters()) + list(projector.parameters()),
        lr=config["pretrain_cpt"]["learning_rate"],
        weight_decay=config["pretrain_cpt"]["weight_decay"],
    )

    grad_accum = max(1, config["pretrain_cpt"].get("gradient_accumulation_steps", 1))

    curriculum_cfg = config["pretrain_cpt"]["curriculum"]
    start_chunks = max(1, curriculum_cfg.get("start_chunks", 1))
    end_chunks = max(start_chunks, curriculum_cfg.get("end_chunks", start_chunks))
    curriculum_steps = max(1, curriculum_cfg.get("steps", 1))

    chunk_char_limit = config["retrieval"].get("chunk_size", 512)
    context_max_tokens = config["encoder"].get("max_length", 256)
    max_seq_len = config["data"].get("max_length", config["llm"].get("max_length", 2048))
    projector_weight = config["pretrain_cpt"].get("projector_weight", 0.2)

    train_samples = config["data"].get("train_samples", None)
    if isinstance(train_samples, int) and train_samples <= 0:
        train_samples = None

    dataloader = create_dataloader(
        config["data"]["path"],
        split="train",
        batch_size=config["pretrain_cpt"]["batch_size"],
        max_samples=train_samples,
        num_workers=config["data"].get("num_workers", 0),
    )

    num_epochs = config["pretrain_cpt"]["num_epochs"]

    print(f"[Stage 2] Continual pretraining for {num_epochs} epochs...")

    global_step = 0
    cumulative_loss = 0.0
    total_updates = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        effective_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            samples = batch_to_samples(batch)

            progress = min(1.0, global_step / curriculum_steps)
            num_expanded = int(start_chunks + (end_chunks - start_chunks) * progress)
            num_expanded = max(1, num_expanded)

            prompts: List[str] = []
            combined_texts: List[str] = []
            prompt_token_lengths: List[int] = []

            for sample in samples:
                expanded, compressed = _split_context_chunks(
                    sample, num_expanded=num_expanded, chunk_char_limit=chunk_char_limit
                )
                prompt = _build_prompt(sample, expanded, compressed)
                answer = str(sample["answer"]).strip()
                full_text = f"{prompt} {answer}"

                prompts.append(prompt)
                combined_texts.append(full_text)

            if not combined_texts:
                global_step += 1
                continue

            tokenized_full = tokenizer(
                combined_texts,
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
            prompt_token_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()

            labels = tokenized_full["input_ids"].clone()
            for idx, prompt_len in enumerate(prompt_token_lengths):
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

            loss = loss / grad_accum
            loss.backward()

            cumulative_loss += loss.item() * grad_accum
            epoch_loss += loss.item() * grad_accum

            if (global_step + 1) % grad_accum == 0:
                clip_grad_norm_(
                    list(llm.parameters()) + list(projector.parameters()),
                    config["pretrain_cpt"]["max_grad_norm"],
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_updates += 1

            effective_batches += 1
            global_step += 1

        avg_epoch_loss = epoch_loss / max(effective_batches, 1)
        print(
            f"[Stage 2] Epoch {epoch+1} | avg loss: {avg_epoch_loss:.4f} | expanded chunks: {num_expanded}"
        )

    avg_loss = cumulative_loss / max(total_updates, 1)
    print(f"[Stage 2] Continual pretraining finished. Average update loss: {avg_loss:.4f}")

    checkpoint = {
        "llm_state_dict": llm.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "epoch": num_epochs,
        "loss": avg_loss,
    }
    save_checkpoint(checkpoint, f"{config['checkpoint_dir']}/pretrain_cpt.pt")
    print(f"[Stage 2] Saved checkpoint to {config['checkpoint_dir']}/pretrain_cpt.pt")


def main() -> None:
    args = tyro.cli(Args)
    train_cpt(args.config)


if __name__ == "__main__":
    main()
