"""Training script for Stage 1: Reconstruction."""
from dataclasses import dataclass
import torch
import tyro
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from refrag.compress.encoder import ChunkEncoder
from refrag.compress.losses import reconstruction_loss
from refrag.compress.projector import Projector
from refrag.data.hotpotqa import create_dataloader
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


def train_reconstruction(config_path: str) -> None:
    """Train reconstruction from compressed vectors to LLM embeddings."""
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    device = get_device(config.get("device", "auto"))

    print(f"[Stage 1] Loading LLM and encoder on {device}...")
    llm, tokenizer = load_llm(
        config["llm"]["model_name"],
        use_peft=config["llm"].get("use_peft", False),
        peft_config=config["llm"].get("peft_config", {}),
        device=str(device),
        torch_dtype=config["llm"].get("torch_dtype", "float16"),
    )
    llm.eval()  # teacher network

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

    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=config["pretrain_recon"]["learning_rate"],
        weight_decay=config["pretrain_recon"]["weight_decay"],
    )

    grad_accum = max(1, config["pretrain_recon"].get("gradient_accumulation_steps", 1))
    context_max_length = config["encoder"].get("max_length", 256)

    train_samples = config["data"].get("train_samples", None)
    if isinstance(train_samples, int) and train_samples <= 0:
        train_samples = None

    dataloader = create_dataloader(
        config["data"]["path"],
        split="train",
        batch_size=config["pretrain_recon"]["batch_size"],
        max_samples=train_samples,
        num_workers=config["data"].get("num_workers", 0),
    )

    projector.train()

    total_steps = 0
    running_loss = 0.0
    total_examples = 0
    num_epochs = config["pretrain_recon"]["num_epochs"]

    print(f"[Stage 1] Starting reconstruction training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        effective_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            samples = batch_to_samples(batch)

            compressed, targets, mask = collate_compression_targets(
                encoder=encoder,
                llm=llm,
                tokenizer=tokenizer,
                samples=samples,
                device=device,
                chunk_char_limit=context_max_length * 4,
                context_max_tokens=context_max_length,
            )

            if compressed.numel() == 0:
                continue

            batch_size, max_chunks, _ = compressed.shape
            compressed_flat = compressed.view(-1, encoder_dim)
            targets_flat = targets.view(-1, llm_dim)
            mask_flat = mask.view(-1)

            logits = projector(compressed_flat)

            loss = reconstruction_loss(logits, targets_flat, mask=mask_flat)
            loss = loss / grad_accum
            loss.backward()

            total_steps += 1
            running_loss += loss.item() * grad_accum

            if total_steps % grad_accum == 0:
                clip_grad_norm_(projector.parameters(), config["pretrain_recon"]["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * grad_accum
            effective_batches += 1
            total_examples += batch_size

        avg_epoch_loss = epoch_loss / max(effective_batches, 1)
        print(f"[Stage 1] Epoch {epoch+1} | avg loss: {avg_epoch_loss:.4f}")

    avg_loss = running_loss / max(total_steps, 1)
    print(f"[Stage 1] Reconstruction training finished. Average loss: {avg_loss:.4f}")

    checkpoint = {
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "epoch": num_epochs,
        "loss": avg_loss,
        "examples": total_examples,
    }
    save_checkpoint(checkpoint, f"{config['checkpoint_dir']}/pretrain_recon.pt")
    print(f"[Stage 1] Saved checkpoint to {config['checkpoint_dir']}/pretrain_recon.pt")


def main() -> None:
    args = tyro.cli(Args)
    train_reconstruction(args.config)


if __name__ == "__main__":
    main()
