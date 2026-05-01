# svomptr_moe/experts.py
import torch
import torch.nn as nn

class GrammarExpert(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts=8):
        super().__init__()
        self.experts = nn.ModuleList([GrammarExpert(hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x):
        # x: [batch, seq, hidden]
        router_logits = self.router(x) # [batch, seq, experts]
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Sparse MoE simplification (Top-1)
        top1_weights, top1_idx = torch.topk(routing_weights, 1, dim=-1)
        
        # Process each expert
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top1_idx == i).squeeze(-1)
            if mask.any():
                out[mask] += expert(x[mask]) * top1_weights[mask]
        return out
