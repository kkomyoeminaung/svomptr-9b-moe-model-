# svomptr_9b/training/phase3_causal_trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer import BaseTrainer

class Phase3CausalTrainer(BaseTrainer):
    """Phase 3: Causal Reasoning training (Contrastive Loss)"""
    
    def __init__(self, model, train_loader, val_loader, config):
        super().__init__(model, train_loader, val_loader, config)
        self.margin = 0.5
        
    def train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc="Phase 3 - Causal Training")
        for batch in pbar:
            pos_logits, neg_logits = self._get_contrastive_logits(batch)
            
            # Contrastive Loss: max(0, margin - (pos - neg))
            loss = torch.mean(torch.clamp(self.margin - (pos_logits - neg_logits), min=0))                
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix({"loss": loss.item()})
            
    def _get_contrastive_logits(self, batch):
        pos_ids = batch['input_ids'].to(self.device)
        # Positive: real sequence
        pos_outputs = self.model(pos_ids)
        # Assuming model returns (logits, ...) or just logits, 
        # based on typical transformer, need to be careful.
        # Looking at original stub: it just needed logits.
        # Re-using the implementation from the repair report.
        if isinstance(pos_outputs, tuple): pos_logits = pos_outputs[0]
        else: pos_logits = pos_outputs
        
        pos_score = pos_logits[:, -1, :].max(dim=-1).values.mean()
        
        # Negative: shuffled sequence
        neg_ids = pos_ids[torch.randperm(pos_ids.size(0))]
        neg_outputs = self.model(neg_ids)
        if isinstance(neg_outputs, tuple): neg_logits = neg_outputs[0]
        else: neg_logits = neg_outputs
            
        neg_score = neg_logits[:, -1, :].max(dim=-1).values.mean()
        return pos_score, neg_score
