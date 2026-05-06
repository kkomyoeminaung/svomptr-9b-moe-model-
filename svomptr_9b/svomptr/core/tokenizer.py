# svomptr/core/tokenizer.py
import torch
from transformers import AutoTokenizer

class RuleTokenizer:
    """A wrapper around pre-trained tokenizer for structural SVOMPTR processing."""
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", vocab_size=50257):
        self._tok = AutoTokenizer.from_pretrained(model_name)
        # Fallback to model's vocab size
        self.vocab_size = vocab_size
        self.pad_token_id = self._tok.pad_token_id or 0
        self.eos_token_id = self._tok.eos_token_id or vocab_size - 1
        
    def __call__(self, text, max_length=128, padding='max_length', truncation=True, **kwargs):
        return self._tok(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            **kwargs
        )
    
    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=True)
