# svomptr_9b/scripts/run_training.py

import argparse
import torch
from svomptr_9b.core.model_9b import SVOMPTR9B, ModelConfig
from svomptr_9b.training.data_builder import DataBuilder
from svomptr_9b.training.phase1_slot_trainer import Phase1SlotTrainer
# ... import other trainers

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    config = ModelConfig()
    model = SVOMPTR9B(config).to("cuda")
    
    # 1. Build Data
    builder = DataBuilder("data/raw/rules.json", "data/training")
    builder.build()
    
    # 2. Run Phases
    # ... logic to run phases in order
    print("🚀 Pipeline Finished!")

if __name__ == "__main__":
    run()
