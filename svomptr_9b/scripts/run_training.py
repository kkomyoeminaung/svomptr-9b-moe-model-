# svomptr_9b/scripts/run_training.py

import argparse
import torch
import os
from svomptr_9b.svomptr.core.model_9b import SVOMPTR9B
from svomptr_9b.svomptr.core.config import ModelConfig
from svomptr_9b.training.data_builder import DataBuilder
from svomptr_9b.training.phase1_slot_trainer import Phase1SlotTrainer
# Ensure other trainers are imported as they are needed
# from svomptr_9b.training.phase2_mlm_trainer import Phase2MLMTrainer
# from svomptr_9b.training.phase3_causal_trainer import Phase3CausalTrainer
# from svomptr_9b.training.phase4_conversation_trainer import Phase4ConversationTrainer

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all", help="1, 2, 3, 4 or all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    config = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SVOMPTR9B(config).to(device)
    
    # 1. Build Data
    raw_data_path = "data/raw/rules.json"
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found.")
        return

    builder = DataBuilder(raw_data_path, "data/training")
    builder.build()
    
    # 2. Run Phases
    phases = [
        ("1", Phase1SlotTrainer, "Phase 1: Slot Logic & Grammar Patterns"),
    ]

    # Mock loaders for now, replace with real ones when implemented
    train_loader = [] 
    val_loader = []

    for phase_id, trainer_class, desc in phases:
        if args.phase in [phase_id, "all"]:
            print(f"\n--- {desc} ---")
            checkpoint_path = f"data/checkpoints/phase_{phase_id}_final.pt"
            
            if args.resume and os.path.exists(checkpoint_path):
                print(f"Resuming from {checkpoint_path}...")
                model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

            trainer = trainer_class(model, train_loader, val_loader, config)
            # trainer.train_epoch() 
        
    print("🚀 Pipeline Finished!")

if __name__ == "__main__":
    run()
