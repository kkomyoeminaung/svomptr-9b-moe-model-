# svomptr/core/model_9b.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import os

# Integration imports
from .config import ModelConfig
from ..layers.quantization import quantize_model
from ..memory.long_term import LongTermMemory
from ..reasoning.thought_chain import ThoughtChain
from ..reasoning.dreamer import Dreamer
from ..layers.virtual_subject import VirtualSubjectDetector, VirtualSubjectEmbedding
from .svomptr_rules import SVOMPTRRuleEngine
from .svomptr_complete import SVOMPTRCompleteParser
from svomptr_moe.experts import MoELayer

class SVOMPTR9B(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        
        # Core Architecture: Sparse MoE Neural Backbone
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Custom MoE-enabled Transformer Blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attention": nn.MultiheadAttention(config.hidden_dim, 8, batch_first=True),
                "norm1": nn.LayerNorm(config.hidden_dim),
                "moe": MoELayer(config.hidden_dim, num_experts=8),
                "norm2": nn.LayerNorm(config.hidden_dim)
            }) for _ in range(config.num_layers)
        ])
        
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.slot_predictor = nn.Linear(config.hidden_dim, 7) # S,V,O,M,P,T,R
        
        # Integrate advanced features
        self.memory = LongTermMemory()
        self.cot = ThoughtChain()
        self.dreamer = Dreamer(self)
        
        # Virtual subject components
        self.virtual_detector = VirtualSubjectDetector()
        self.virtual_embedding = VirtualSubjectEmbedding(config.hidden_dim)
        
        # SVOMPTR Rule Engine
        self.rule_engine = SVOMPTRRuleEngine()
        self.complete_parser = SVOMPTRCompleteParser()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_slots: bool = False):
        x = self.embedding(input_ids)
        
        # Deep Sparse MoE processing
        for block in self.blocks:
            # Self-Attention (support attention_mask if provided)
            # Support for PyTorch MultiheadAttention key_padding_mask
            key_padding_mask = None
            if attention_mask is not None:
                # Transformers attention_mask is (batch, seq), 1 for keep, 0 for mask
                # PyTorch key_padding_mask is (batch, seq), False for keep, True for mask
                key_padding_mask = (attention_mask == 0)
                
            attn_out, _ = block["attention"](x, x, x, key_padding_mask=key_padding_mask)
            x = block["norm1"](x + attn_out)
            
            # Sparse Expert Routing
            moe_out = block["moe"](x)
            x = block["norm2"](x + moe_out)
            
        logits = self.lm_head(x)
        
        if return_slots:
            slot_logits = self.slot_predictor(x)
            return logits, x, slot_logits
            
        return logits, x

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 50):
        """Real auto-regressive generation logic."""
        self.eval()
        generated = prompt_ids
        eos_token_id = getattr(self.config, 'eos_token_id', self.config.vocab_size - 1)                

        eos_ids = [eos_token_id]
        if hasattr(self.config, 'pad_token_id'):
            eos_ids.append(self.config.pad_token_id)
        
        for _ in range(max_new_tokens):
            logits, _ = self.forward(generated[:, -512:]) # Context window
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() in eos_ids: # EOS check
                break
        return generated

    def chat(self, message: str, tokenizer=None) -> Dict:
        """Hybrid approach: Rules + Neural Generation"""
        # Step 1: Structural Extraction (Rule-based)
        frame = self.rule_engine.parse(message)
        
        # Update Virtual Subject Context
        self.virtual_detector.update_context(frame.S, frame.O if "There" in message else None)
        
        # Step 2: Neural Response Generation & Slot Prediction
        neural_slots = None
        if tokenizer:
            tokens = tokenizer(message, return_tensors="pt").to(self.embedding.weight.device)
            input_ids = tokens['input_ids'] if isinstance(tokens, dict) else tokens
            
            with torch.no_grad():
                # Get both text and slots Neurallly
                logits, hidden, slot_logits = self.forward(input_ids, return_slots=True)
                generated = self.generate(input_ids)
                
            response_text = tokenizer.decode(generated[0])
            
            # Extract neural slot IDs (argmax)
            slot_ids = torch.argmax(slot_logits, dim=-1)[0].tolist()
            SLOT_ORDER = ['S', 'V', 'O', 'M', 'P', 'T', 'R']
            neural_slots = [SLOT_ORDER[s] for s in slot_ids if s < len(SLOT_ORDER)]
        else:
            response_text = f"Analyzed Sentence: S={frame.S}, V={frame.V}. I am ready to process the grammar."
        
        ui_frame = {
            "Subject": frame.S or "Inferred",
            "Verb": frame.V or "Action",
            "Object": frame.O or "Target",
            "Modifier": frame.M or "Detail",
            "Place": frame.P or "Loc",
            "Time": frame.T or "Time",
            "Reason": frame.R or "Cause",
            "Neural_Confidence": neural_slots[:5] if neural_slots else "Rules-Only"
        }
        
        return {"response": response_text, "frame": ui_frame}

        
    def idle(self):
        """Invoke this when no user interaction occurs."""
        self.dreamer.dream()

    def clear_chat_context(self):
        """Reset virtual subject and memory context"""
        self.virtual_detector.clear_context()
        self.memory.clear()
        print("🧹 SVOMPTR Context Cleared.")
