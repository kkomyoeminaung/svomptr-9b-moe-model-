
from dataclasses import dataclass

@dataclass
class MoEConfig:
    hidden_dim: int = 1024
    num_experts: int = 8
    top_k: int = 2
    router_dim: int = 384  # MiniLM embedding size
    expert_domains: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_steps: int = 1000
