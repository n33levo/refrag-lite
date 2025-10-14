"""LLM loading."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Dict, Any

def load_llm(model_name: str, use_peft: bool = False, peft_config: Optional[Dict[str, Any]] = None,
            device: str = "cuda", torch_dtype: str = "bfloat16") -> tuple:
    dtype = getattr(torch, torch_dtype) if torch_dtype != "float32" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device, trust_remote_code=True
    )

    if use_peft and peft_config:
        lora_config = LoraConfig(
            r=peft_config.get("lora_r", 16),
            lora_alpha=peft_config.get("lora_alpha", 32),
            lora_dropout=peft_config.get("lora_dropout", 0.05),
            target_modules=peft_config.get("target_modules", ["q_proj", "v_proj"]),
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer