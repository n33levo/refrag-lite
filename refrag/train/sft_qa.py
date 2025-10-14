"""Training script for Stage 3: Supervised Fine-Tuning."""
import torch
import tyro
from dataclasses import dataclass
from refrag.llm.model import load_llm
from refrag.compress.encoder import ChunkEncoder
from refrag.compress.projector import Projector
from refrag.utils.io import load_yaml as load_config, get_device, save_checkpoint


@dataclass
class Args:
    config: str = "configs/default.yaml"


def train_sft(config_path: str):
    """Train with SFT using QA pairs and mixed contexts."""
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
        lr=config["sft"]["learning_rate"],
        weight_decay=config["sft"]["weight_decay"]
    )
    
    # Training loop for QA with mixed inputs
    print(f"Starting SFT training for {config['sft']['num_epochs']} epochs...")
    
    num_samples = config["data"]["train_samples"]
    batch_size = config["sft"]["batch_size"]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Fixed compression/expansion ratios
    num_compressed = config["sft"]["num_compressed"]
    num_expanded = config["sft"]["num_expanded"]
    
    llm.train()
    projector.train()
    total_loss = 0.0
    
    for epoch in range(config['sft']['num_epochs']):
        epoch_loss = 0.0
        
        for batch_idx in range(num_batches):
            batch_size_actual = min(batch_size, num_samples - batch_idx * batch_size)
            
            # Create dummy QA data
            # In real implementation: load HotpotQA samples
            query_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size_actual, 30), device=device)
            answer_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size_actual, 20), device=device)
            
            # Create mixed context: query + compressed chunks + expanded chunks + answer
            # Compressed chunks (as vectors that will be projected)
            compressed_vecs = torch.randn(batch_size_actual, num_compressed, encoder_dim, device=device)
            projected_vecs = projector(compressed_vecs.view(-1, encoder_dim))
            projected_vecs = projected_vecs.view(batch_size_actual, num_compressed, llm_dim)
            
            # Expanded chunks (as token IDs)
            expanded_chunks = torch.randint(0, tokenizer.vocab_size, (batch_size_actual, num_expanded, 100), device=device)
            
            # Create input sequence
            # Format: [CLS] query [SEP] compressed_vectors [SEP] expanded_chunks [SEP] answer
            input_ids = torch.cat([
                query_tokens,
                torch.full((batch_size_actual, 1), tokenizer.eos_token_id, device=device),  # SEP
                expanded_chunks.view(batch_size_actual, -1),
                torch.full((batch_size_actual, 1), tokenizer.eos_token_id, device=device),  # SEP
                answer_tokens
            ], dim=1)
            
            # Truncate to max length
            max_length = config["llm"]["max_length"]
            if input_ids.size(1) > max_length:
                input_ids = input_ids[:, :max_length]
            
            # Create labels for next token prediction (only on answer part)
            labels = input_ids.clone()
            # Don't compute loss on query and context
            labels[:, :query_tokens.size(1) + 1 + expanded_chunks.size(1) + 1] = -100
            labels[:, -1] = -100  # Ignore last token
            
            # Forward pass
            outputs = llm(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Add regularization loss for compressed vectors
            # Encourage compressed vectors to be meaningful
            compressed_reg_loss = torch.mean(torch.norm(projected_vecs, dim=-1))
            loss += 0.01 * compressed_reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm.parameters(), config['sft']['max_grad_norm'])
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{config['sft']['num_epochs']}, "
                      f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    avg_total_loss = total_loss / (config['sft']['num_epochs'] * num_batches)
    print(f"SFT training completed. Average loss: {avg_total_loss:.4f}")
    
    # Save checkpoint
    checkpoint = {
        "llm_state_dict": llm.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "epoch": config['sft']['num_epochs'],
        "loss": avg_total_loss
    }
    save_checkpoint(checkpoint, f"{config['checkpoint_dir']}/sft_qa.pt")
    print(f"Checkpoint saved to {config['checkpoint_dir']}/sft_qa.pt")


def main():
    args = tyro.cli(Args)
    train_sft(args.config)

if __name__ == "__main__":
    main()
