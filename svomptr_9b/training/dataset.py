# svomptr_9b/training/dataset.py

import os
import json
import torch
from torch.utils.data import Dataset

class SVOMPTRDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.samples.append(json.loads(line))
            except Exception as e:
                print(f"❌ Error loading dataset from {file_path}: {e}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize Input
        inputs = self.tokenizer(
            sample.get('en', sample.get('input', '')),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Robust access to input_ids and attention_mask
        input_ids = inputs.input_ids if hasattr(inputs, 'input_ids') else inputs['input_ids']
        attention_mask = inputs.attention_mask if hasattr(inputs, 'attention_mask') else inputs['attention_mask']
        
        # Convert slots/targets to labels (simplification for Phase 1)
        target = sample.get('target', '')
        slots = torch.zeros(self.max_length, dtype=torch.long)
        SLOT_MAP = {'S': 0, 'V': 1, 'O': 2, 'M': 3, 'P': 4, 'T': 5, 'R': 6}
        
        if isinstance(target, str):
            # Handle format: S(text)-V(text)-O(text)
            if '-' in target:
                parts = target.split('-')
                for i, part in enumerate(parts[:self.max_length]):
                    if '(' in part:
                        slot_key = part.split('(')[0]
                        if slot_key in SLOT_MAP:
                            slots[i] = SLOT_MAP[slot_key]
            else:
                # Fallback to space-separated
                for i, slot_char in enumerate(target.split()[:self.max_length]):
                    slot_key = slot_char.split(':')[0] if ':' in slot_char else slot_char
                    if slot_key in SLOT_MAP:
                        slots[i] = SLOT_MAP[slot_key]
        elif isinstance(target, dict):
            # SVOMPTR dict format: {S: ..., V: ..., O: ...}
            slot_order = ['S', 'V', 'O', 'M', 'P', 'T', 'R']
            for i, s in enumerate(slot_order):
                if target.get(s) and i < self.max_length:
                    slots[i] = SLOT_MAP[s]
                    
        # Mask padding in labels
        labels = input_ids.squeeze(0).clone()
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        labels[labels == pad_id] = -100
        
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "slots": slots,
            "labels": labels
        }
