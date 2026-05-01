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
        
        # Convert slots/targets to labels (simplification for Phase 1)
        # In a real scenario, this would map SVOMPTR strings to class indices
        slots = torch.zeros(self.max_length, dtype=torch.long)
        
        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "slots": slots
        }
