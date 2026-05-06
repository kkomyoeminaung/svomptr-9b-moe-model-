# svomptr_9b/training/phase2_mlm_trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer import BaseTrainer

class Phase2MLMTrainer(BaseTrainer):
    """Phase 2: Masked Language Model training"""
    
    def __init__(self, model, train_loader, val_loader, config):
        super().__init__(model, train_loader, val_loader, config)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc="Phase 2 - MLM Training")
        for batch in pbar:
            self.optimizer.zero_grad()
            loss = self.train_step(batch)
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    def train_step(self, batch):
        inputs, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
        logits, _ = self.model(inputs)
        return self.criterion(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
