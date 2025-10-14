"""Training script for Stage 2: Continual Pretraining."""
import torch
import tyro
from dataclasses import dataclass
from refrag.llm.model import load_llm
from refrag.compress.encoder import ChunkEncoder
from refrag.compress.projector import Projector
from refrag.compress.losses import reconstruction_loss
from refrag.utils.io import load_yaml as load_config, get_device, save_checkpoint


@dataclass
class Args:
    config: str = "configs/default.yaml"


def train_cpt(config_path: str):
    """Train with continual pretraining using mixed contexts."""
    config = load_config(config_path)
    
    # Set device
    device = get_device(config.get("device", "auto"))
    
    # Load models
    print(f"Loading models on {device}...")
    llm, tokenizer = load_llm(
        config["llm"]["model_name"],
        use_peft=config["llm"]["use_peft"],
        peft_config=config["llm"]["peft_config"],
        device=device,
        torch_dtype=config["llm"]["torch_dtype"]
    )
    
    encoder = ChunkEncoder(config["encoder"]["model_name"], device=device)
    
    # Load projector
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
        list(llm.parameters()) + list(projector.parameters()),
        lr=config["pretrain_cpt"]["learning_rate"],
        weight_decay=config["pretrain_cpt"]["weight_decay"]
    )
    
    # Training loop with curriculum learning
    print(f"Starting continual pretraining for {config['pretrain_cpt']['num_epochs']} epochs...")
    
    # Curriculum learning setup
    curriculum = config['pretrain_cpt']['curriculum']
    start_chunks = curriculum['start_chunks']
    end_chunks = curriculum['end_chunks']
    curriculum_steps = curriculum['steps']
    
    num_samples = config["data"]["train_samples"]
    batch_size = config["pretrain_cpt"]["batch_size"]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    llm.train()
    projector.train()
    total_loss = 0.0
    
    for epoch in range(config['pretrain_cpt']['num_epochs']):
        epoch_loss = 0.0
        
        for batch_idx in range(num_batches):
            # Curriculum: gradually increase number of chunks
            progress = (epoch * num_batches + batch_idx) / curriculum_steps
            progress = min(progress, 1.0)
            num_chunks = int(start_chunks + (end_chunks - start_chunks) * progress)
            
            batch_size_actual = min(batch_size, num_samples - batch_idx * batch_size)
            
            # Create dummy mixed context data
            # In real implementation: load text chunks, some compressed, some expanded
            query_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size_actual, 20), device=device)
            
            # Compressed chunks (as vectors)
            compressed_chunks = torch.randn(batch_size_actual, num_chunks, llm_dim, device=device)
            
            # Expanded chunks (as token IDs)
            expanded_chunks = torch.randint(0, tokenizer.vocab_size, (batch_size_actual, num_chunks, 50), device=device)
            
            # Create input sequence: query + compressed + expanded
            # This is a simplified version - real implementation would use proper tokenization
            input_ids = torch.cat([
                query_tokens,
                torch.randint(0, tokenizer.vocab_size, (batch_size_actual, 1), device=device),  # separator
                expanded_chunks.view(batch_size_actual, -1)
            ], dim=1)
            
            # Truncate to max length
            max_length = config["llm"]["max_length"]
            if input_ids.size(1) > max_length:
                input_ids = input_ids[:, :max_length]
            
            # Create labels for next token prediction
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore last token
            
            # Forward pass
            outputs = llm(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Add projector loss (reconstruction from compressed chunks)
            if num_chunks > 0:
                # Dummy compressed vectors
                compressed_vecs = torch.randn(batch_size_actual, num_chunks, encoder_dim, device=device)
                projected_vecs = projector(compressed_vecs.view(-1, encoder_dim))
                target_embeddings = torch.randn(batch_size_actual * num_chunks, llm_dim, device=device)
                projector_loss = reconstruction_loss(projected_vecs, target_embeddings)
                loss += 0.1 * projector_loss  # Weight the projector loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm.parameters(), config['pretrain_cpt']['max_grad_norm'])
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{config['pretrain_cpt']['num_epochs']}, "
                      f"Batch {batch_idx}/{num_batches}, Chunks: {num_chunks}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    avg_total_loss = total_loss / (config['pretrain_cpt']['num_epochs'] * num_batches)
    print(f"Continual pretraining completed. Average loss: {avg_total_loss:.4f}")
    
    # Save checkpoint
    checkpoint = {
        "llm_state_dict": llm.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "epoch": config['pretrain_cpt']['num_epochs'],
        "loss": avg_total_loss
    }
    save_checkpoint(checkpoint, f"{config['checkpoint_dir']}/pretrain_cpt.pt")
    print(f"Checkpoint saved to {config['checkpoint_dir']}/pretrain_cpt.pt")


def main():
    args = tyro.cli(Args)
    train_cpt(args.config)

if __name__ == "__main__":
    main()
