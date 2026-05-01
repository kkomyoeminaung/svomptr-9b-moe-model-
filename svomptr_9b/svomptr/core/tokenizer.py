# svomptr/core/tokenizer.py
import torch

class RuleTokenizer:
    """A rule-based tokenizer for structural SVOMPTR processing."""
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = vocab_size - 1
        
    def __call__(self, text, max_length=128, padding='max_length', truncation=True, **kwargs):
        # Extremely simple deterministic mapping for preview
        ids = [ord(c) % (self.vocab_size - 2) + 1 for c in str(text)[:max_length]]
        
        if truncation:
            ids = ids[:max_length]
            
        attention_mask = [1] * len(ids)
        
        if padding == 'max_length':
            pad_len = max_length - len(ids)
            ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long)
        }
    
    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join([chr(i - 1) if i > 0 else "" for i in ids if i < self.vocab_size - 1])
