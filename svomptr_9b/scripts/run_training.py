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
    
    from svomptr_9b.training.dataset import SVOMPTRDataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 1. Build Data
    raw_data_path = "data/raw/rules.json"
    processed_dir = "data/training"
    builder = DataBuilder(raw_data_path, processed_dir)
    builder.build()
    
    # 2. Setup Loaders
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # Use any standard tokenizer for testing
    phase1_data = os.path.join(processed_dir, "phase1.jsonl")
    
    if os.path.exists(phase1_data):
        print(f"📊 Loading Phase 1 data from {phase1_data}")
        train_ds = SVOMPTRDataset(phase1_data, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size if hasattr(config, 'batch_size') else 4, shuffle=True)
    else:
        print(f"⚠️ Warning: {phase1_data} not found. Ensure raw data exists.")
        train_loader = []
    
    val_loader = [] # Placeholder
    
    # 3. Run Phases
    phases = [
        ("1", Phase1SlotTrainer, "Phase 1: Slot Logic & Grammar Patterns"),
        # ("2", Phase2MLMTrainer, "Phase 2: Bilingual Masked Modeling"), # Future
        # ("3", Phase3CausalTrainer, "Phase 3: Domain Pretraining"), # Future
        # ("4", Phase4ConversationTrainer, "Phase 4: Bilingual SFT + CoT"), # Future
    ]

    # Real loaders should be initialized here
    # train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    train_loader = [] # Placeholder
    val_loader = [] # Placeholder

    for phase_id, trainer_class, desc in phases:
        if args.phase in [phase_id, "all"]:
            print(f"\n{'='*20}")
            print(f"🚀 Starting {desc}")
            print(f"{'='*20}")
            
            checkpoint_dir = f"data/checkpoints/phase_{phase_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            latest_checkpoint = os.path.join(checkpoint_dir, "latest.pt")
            
            if args.resume and os.path.exists(latest_checkpoint):
                print(f"♻️  Resuming from {latest_checkpoint}...")
                model.load_state_dict(torch.load(latest_checkpoint)['model_state_dict'])

            trainer = trainer_class(model, train_loader, val_loader, config)
            
            epochs = config.num_epochs if hasattr(config, 'num_epochs') else 3
            for epoch in range(epochs):
                print(f"\n📅 Epoch {epoch+1}/{epochs}")
                trainer.train_epoch()
                
                # Save epoch checkpoint
                epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
                trainer.save_checkpoint(epoch_path)
                trainer.save_checkpoint(latest_checkpoint) # Always update latest
                
                # trainer.validate()
        
    print("\n✅ All Training Phases Completed Successfully. Pipeline Finished!")

if __name__ == "__main__":
    run()
