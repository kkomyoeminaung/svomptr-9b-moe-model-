# ml_model/svomptr/core/config.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml

@dataclass
class ModelConfig:
    model_name: str = "svomptr-grand-3b"
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_slots: int = 7
    vocab_size: int = 100000
    max_seq_len: int = 4096
    dropout: float = 0.1
    slot_predictor_layers: int = 3
    max_recursion_depth: int = 5
    working_memory_size: int = 10
    memory_embed_dim: int = 384
    long_term_db_path: str = "data/database/prd.db"
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 50
    default_repetition_penalty: float = 1.1
    default_max_new_tokens: int = 256
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    use_fp16: bool = True
    use_bf16: bool = False
    gradient_checkpointing: bool = True
    quantization: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data["model"])

    def to_dict(self) -> dict:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)
