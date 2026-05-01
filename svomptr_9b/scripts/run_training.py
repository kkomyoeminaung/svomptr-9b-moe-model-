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
    tokenizer = AutoTokenizer.from_pretrained("gpt2") 
    phase1_data = os.path.join(processed_dir, "phase1.jsonl")
    
    train_loader = []
    if os.path.exists(phase1_data):
        print(f"📊 Loading Phase 1 data from {phase1_data}")
        train_ds = SVOMPTRDataset(phase1_data, tokenizer)
        if len(train_ds) > 0:
            train_loader = DataLoader(train_ds, batch_size=config.batch_size if hasattr(config, 'batch_size') else 4, shuffle=True)
            print(f"✅ Loaded {len(train_ds)} samples for Phase 1.")
        else:
            print(f"⚠️ Warning: {phase1_data} is empty.")
    else:
        print(f"⚠️ Warning: {phase1_data} not found. Ensure raw data exists.")
    
    val_loader = [] 
    
    # 3. Run Phases
    phases = [
        ("1", Phase1SlotTrainer, "Phase 1: Slot Logic & Grammar Patterns"),
    ]

    for phase_id, trainer_class, desc in phases:
        if args.phase in [phase_id, "all"]:
            print(f"\n{'='*20}")
            print(f"🚀 Starting {desc}")
            print(f"{'='*20}")
            
            checkpoint_dir = f"data/checkpoints/phase_{phase_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            latest_checkpoint = os.path.join(checkpoint_dir, "latest.pt")
            start_epoch = 0
            
            if args.resume and os.path.exists(latest_checkpoint):
                print(f"♻️  Resuming from {latest_checkpoint}...")
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"📈 Resuming from Epoch {start_epoch + 1}")

            trainer = trainer_class(model, train_loader, val_loader, config)
            if args.resume and os.path.exists(latest_checkpoint):
                # Optionally restore optimizer state
                try:
                    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except: pass
            
            epochs = config.num_epochs if hasattr(config, 'num_epochs') else 3
            for epoch in range(start_epoch, epochs):
                print(f"\n📅 Epoch {epoch+1}/{epochs}")
                trainer.train_epoch()
                
                # Save epoch checkpoint
                epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
                trainer.save_checkpoint(epoch_path)
                
                # Save with metadata for resume
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                }, latest_checkpoint)
                print(f"💾 Checkpoint updated at {latest_checkpoint}")
                
                # trainer.validate()
        
    print("\n✅ All Training Phases Completed Successfully. Pipeline Finished!")

if __name__ == "__main__":
    run()
