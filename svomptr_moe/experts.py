
import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts=8):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_dim, num_experts)
        
    def forward(self, x):
        # gate_scores shape: (batch, seq, num_experts)
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        
        # Only use top-2 experts for efficiency
        top_k_values, top_k_indices = torch.topk(gate_scores, k=2, dim=-1)
        
        # Normalize top-k weights
        top_k_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)
        
        batch_size, seq_len, _ = x.shape
        out = torch.zeros_like(x)
        
        # This is a simplified MoE forward pass for the 1.5B/9B context
        for i in range(2):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_values[:, :, i].unsqueeze(-1)
            
            # Since we have small number of experts, we can iterate
            # In a very large scale, we'd use scatter/gather
            for e_idx in range(len(self.experts)):
                mask = (expert_idx == e_idx)
                if mask.any():
                    # Flattened indexing for expert routing
                    expert_input = x[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    # Ensure weight is (N, 1) for broadcasting over (N, hidden)
                    w = expert_weight[mask].view(-1, 1)
                    out[mask] += expert_output * w
                    
        return out
