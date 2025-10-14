"""LLM inference."""
import torch
from typing import List, Optional

def generate_answer(model, tokenizer, input_embeds: torch.Tensor, attention_mask: torch.Tensor,
                   max_new_tokens: int = 100, temperature: float = 0.7) -> List[str]:
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def compute_perplexity(model, input_embeds: torch.Tensor, attention_mask: torch.Tensor,
                      labels: Optional[torch.Tensor] = None) -> float:
    with torch.no_grad():
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        return torch.exp(outputs.loss).item()