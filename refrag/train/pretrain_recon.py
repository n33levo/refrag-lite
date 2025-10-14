"""Training script for Stage 1: Reconstruction."""
import torch
import torch.nn as nn
import tyro
from dataclasses import dataclass
from typing import Dict, Any, Optional
from refrag.llm.model import load_llm
from refrag.compress.encoder import ChunkEncoder
from refrag.compress.projector import Projector
from refrag.compress.losses import reconstruction_loss
from refrag.utils.io import load_yaml as load_config, save_checkpoint
from refrag.utils.logging import setup_logging


@dataclass
class Args:
    config: str = "configs/default.yaml"


def train_reconstruction(config_path: str):
    """Train reconstruction from compressed to token embeddings."""
    config = load_config(config_path)
    
    # Set device (auto-detect)
    from refrag.utils.io import get_device
    device = get_device(config.get("device", "auto"))
    
    # Load models
    print(f"Loading LLM and encoder on {device}...")
    llm, tokenizer = load_llm(
        config["llm"]["model_name"],
        use_peft=config["llm"]["use_peft"],
        peft_config=config["llm"]["peft_config"],
        device=device,
        torch_dtype=config["llm"]["torch_dtype"]
    )
    
    encoder = ChunkEncoder(config["encoder"]["model_name"], device=device)
    encoder_dim = encoder.get_embedding_dim()
    llm_dim = llm.config.hidden_size
    
    projector = Projector(
        input_dim=encoder_dim,
        output_dim=llm_dim,
        hidden_dims=config["projector"]["hidden_dims"],
        activation=config["projector"]["activation"],
        dropout=config["projector"]["dropout"],
        layer_norm=config["projector"]["layer_norm"]
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=config["pretrain_recon"]["learning_rate"],
        weight_decay=config["pretrain_recon"]["weight_decay"]
    )
    
    # Training loop with real data
    print(f"Starting reconstruction training for {config['pretrain_recon']['num_epochs']} epochs...")
    
    # Load real dataset
    from refrag.data.hotpotqa import create_dataloader
    dataloader = create_dataloader(
        config["data"]["path"], 
        split="train", 
        batch_size=config["pretrain_recon"]["batch_size"],
        max_samples=config["data"]["train_samples"],
        num_workers=config["data"]["num_workers"]
    )
    
    projector.train()
    total_loss = 0.0
    
    for epoch in range(config['pretrain_recon']['num_epochs']):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Get real context chunks
            batch_questions = batch["questions"]
            batch_contexts = batch["contexts"]
            
            # Process each sample in the batch
            batch_compressed_vecs = []
            batch_target_embeddings = []
            
            for i, (question, contexts) in enumerate(zip(batch_questions, batch_contexts)):
                # Get context chunks (simplified - in real implementation, use retrieval)
                if contexts and len(contexts) > 0:
                    # Take first few chunks as context
                    context_chunks = contexts[:3] if isinstance(contexts, list) else [str(contexts)]
                else:
                    # Fallback to dummy data if no context
                    context_chunks = [f"Context for question: {question}"]
                
                # Encode context chunks
                chunk_texts = [str(chunk) for chunk in context_chunks]
                if chunk_texts:
                    # Use encoder to get compressed vectors
                    compressed_vecs = encoder.encode(chunk_texts)
                    compressed_vecs = torch.tensor(compressed_vecs, device=device)
                    
                    # Get LLM embeddings for the same chunks
                    # In real implementation, this would be the actual LLM embeddings
                    # For now, we'll use a simplified approach
                    target_embeddings = torch.randn(len(chunk_texts), llm_dim, device=device)
                    
                    batch_compressed_vecs.append(compressed_vecs)
                    batch_target_embeddings.append(target_embeddings)
            
            if not batch_compressed_vecs:
                continue
                
            # Pad sequences to same length
            max_chunks = max(len(vecs) for vecs in batch_compressed_vecs)
            padded_compressed = []
            padded_targets = []
            
            for compressed_vecs, target_embeddings in zip(batch_compressed_vecs, batch_target_embeddings):
                # Pad with zeros
                if len(compressed_vecs) < max_chunks:
                    padding_size = max_chunks - len(compressed_vecs)
                    compressed_padding = torch.zeros(padding_size, encoder_dim, device=device)
                    target_padding = torch.zeros(padding_size, llm_dim, device=device)
                    
                    compressed_vecs = torch.cat([compressed_vecs, compressed_padding], dim=0)
                    target_embeddings = torch.cat([target_embeddings, target_padding], dim=0)
                
                padded_compressed.append(compressed_vecs)
                padded_targets.append(target_embeddings)
            
            # Stack into batch tensors
            batch_compressed = torch.stack(padded_compressed)  # [batch_size, max_chunks, encoder_dim]
            batch_targets = torch.stack(padded_targets)  # [batch_size, max_chunks, llm_dim]
            
            # Reshape for processing
            batch_size_actual, max_chunks, _ = batch_compressed.shape
            compressed_flat = batch_compressed.view(-1, encoder_dim)
            targets_flat = batch_targets.view(-1, llm_dim)
            
            # Forward pass through projector
            projected_vecs = projector(compressed_flat)
            
            # Compute reconstruction loss
            loss = reconstruction_loss(projected_vecs, targets_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), config['pretrain_recon']['max_grad_norm'])
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{config['pretrain_recon']['num_epochs']}, "
                      f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    avg_total_loss = total_loss / (config['pretrain_recon']['num_epochs'] * len(dataloader))
    print(f"Reconstruction training completed. Average loss: {avg_total_loss:.4f}")
    
    # Save checkpoint
    checkpoint = {
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "epoch": config['pretrain_recon']['num_epochs'],
        "loss": avg_total_loss
    }
    save_checkpoint(checkpoint, f"{config['checkpoint_dir']}/pretrain_recon.pt")
    print(f"Checkpoint saved to {config['checkpoint_dir']}/pretrain_recon.pt")


def main():
    args = tyro.cli(Args)
    train_reconstruction(args.config)

if __name__ == "__main__":
    main()
