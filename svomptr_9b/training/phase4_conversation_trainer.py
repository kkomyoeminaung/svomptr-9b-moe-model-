# svomptr_9b/training/phase4_conversation_trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer import BaseTrainer

class Phase4ConversationTrainer(BaseTrainer):
    """Phase 4: Conversation Fine-Tuning"""
    
    def __init__(self, model, train_loader, val_loader, config):
        super().__init__(model, train_loader, val_loader, config)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc="Phase 4 - Chat SFT")
        for batch in pbar:
            inputs, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(inputs)
            
            loss = self.criterion(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
            loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix({"loss": loss.item()})
