import torch
from typing import List

from dataclasses import dataclass, field
from typing import List

@dataclass
class MoEConfig:
    num_experts: int = 13
    chat_expert_size: str = "2.5B"  # Main expert (Always Active)
    sub_expert_size: str = "0.5B"   # Each sub-expert (On-demand)
    always_active_experts: List[int] = field(default_factory=lambda: [0])  # Expert 0 is Chat
    experts_per_token: int = 2  # Chat + Top-1 Sub-expert
    router_threshold: float = 0.7  # Confidence threshold for routing
    
    hidden_dim: int = 2048 # Base hidden dimension
    
    domain_names: List[str] = field(default_factory=lambda: [
        "chat",        # MAIN - Expert 0
        "software",    # Expert 1
        "medicine",    # Expert 2
        "engineering", # Expert 3
        "buddhism",    # Expert 4
        "history",     # Expert 5
        "science",     # Expert 6
        "cosmology",   # Expert 7
        "politics",    # Expert 8
        "economics",   # Expert 9
        "psychology",  # Expert 10
        "philosophy",  # Expert 11
        "art"          # Expert 12
    ])
    
    expert_sizes_mb: List[int] = field(default_factory=lambda: [
        2500,  # chat
        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500
    ])
