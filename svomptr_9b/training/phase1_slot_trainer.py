# svomptr_9b/training/phase1_slot_trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer import BaseTrainer

class Phase1SlotTrainer(BaseTrainer):
    """Phase 1: Slot Predictor training"""
    
    def __init__(self, model, train_loader, val_loader, config):
        super().__init__(model, train_loader, val_loader, config)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc="Phase 1 - Training")
        for batch in pbar:
            inputs, target_slots = batch['input_ids'].to(self.device), batch['slots'].to(self.device)
            
            self.optimizer.zero_grad()
            _, _, slot_logits = self.model(inputs, return_slots=True)
            
            loss = self.criterion(slot_logits.view(-1, 7), target_slots.view(-1))
            loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix({"loss": loss.item()})
