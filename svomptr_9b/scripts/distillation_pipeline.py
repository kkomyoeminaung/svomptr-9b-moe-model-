# Knowledge Distillation Pipeline for SVOMPTR-9B
# Uses Qwen-7B as a teacher to train SVOMPTR-9B student
# Supports checkpointing and resuming training

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm
from pathlib import Path

# 1. Setup Brain Directory
# Use environment variable for BASE_DIR to avoid hardcoded paths
BASE_DIR = os.environ.get('SVOMPTR_BRAIN_PATH', './svomptr_brain')
CHECKPOINT_FILE = os.path.join(BASE_DIR, 'distillation_checkpoint.json')
os.makedirs(BASE_DIR, exist_ok=True)
print(f"✅ Brain storage set at {BASE_DIR}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"epoch": 0, "step": 0}

def save_checkpoint(epoch, step, optimizer=None, model=None):
    data = {"epoch": epoch, "step": step}
    if optimizer:
        torch.save(optimizer.state_dict(), os.path.join(BASE_DIR, "optim_state.pt"))
    if model:
        torch.save(model.state_dict(), os.path.join(BASE_DIR, f"model_ep{epoch}_step{step}.pt"))
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f"💾 Checkpoint saved: Epoch {epoch}, Step {step}")

# 2. Distillation Trainer
class DistillationTrainer:
    def __init__(self, teacher_model_name, student_model):
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        # Distillation uses CPU/GPU appropriately
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_name, 
            device_map="auto" if self.device=="cuda" else None, 
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            trust_remote_code=True
        )
        self.student = student_model.to(self.device)
        self.student_vocab_size = student_model.config.vocab_size
        self.teacher.eval() # Teacher is frozen
        self.student.train()
        
    def distillation_loss(self, student_logits, teacher_logits, temperature=2.0):
        # KL Divergence between soft targets
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    def train_step(self, batch_inputs):
        batch_inputs = batch_inputs.to(self.device)
        with torch.no_grad():
            teacher_logits = self.teacher(batch_inputs).logits
            teacher_logits = teacher_logits[:, :, :self.student_vocab_size]
        
        # Fixed Bug #5: Student model now has forward() implemented
        student_logits, _ = self.student(batch_inputs)
        
        loss = self.distillation_loss(student_logits, teacher_logits, temperature=self.student.config.distillation_temperature if hasattr(self.student, 'config') and hasattr(self.student.config, 'distillation_temperature') else 2.0)
        return loss

# 3. Main Data Distillation Execution
def run_distillation(total_epochs=5):
    from svomptr_9b.svomptr.core.model_9b import SVOMPTR9B
    from svomptr_9b.svomptr.core.config import ModelConfig
    
    config = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Health Check (Storage)
    print(f"🔍 Validating storage at {BASE_DIR}...")
    try:
        test_file = os.path.join(BASE_DIR, ".write_sync_test")
        with open(test_file, "w") as f: 
            f.write("sync_ok")
            f.flush()
            os.fsync(f.fileno())
        os.remove(test_file)
        print("✅ Storage verification passed.")
    except Exception as e:
        print(f"❌ CRITICAL: Brain storage ({BASE_DIR}) is not writable!")
        print("⚠️ If you are using Google Drive, make sure it is mounted correctly.")
        return

    student_model = SVOMPTR9B(config).to(device)
    
    checkpoint = load_checkpoint()
    start_epoch = checkpoint["epoch"]
    
    if start_epoch >= total_epochs:
        print(f"✅ Distillation already completed for {total_epochs} epochs.")
        return

    print(f"🚀 Distillation Starting (Resume from Epoch {start_epoch + 1})")
    
    trainer = DistillationTrainer(config.TEACHER_MODEL, student_model)
    optimizer = torch.optim.AdamW(trainer.student.parameters(), lr=1e-5)
    
    # Dataset check
    data_source = "data/raw/distillation_sources.json"
    if not os.path.exists(data_source):
        print(f"💡 Seed data missing. Generating 100 synthetic training batches...")
        train_loader = [torch.randint(0, config.vocab_size, (2, 128)) for _ in range(100)]
    else:
        from torch.utils.data import DataLoader
        from svomptr_9b.training.dataset import SVOMPTRDataset
        from svomptr_9b.svomptr.core.tokenizer import RuleTokenizer
        tok = RuleTokenizer(vocab_size=config.vocab_size)
        ds = SVOMPTRDataset(data_source, tok)
        train_loader = DataLoader(ds, batch_size=2, shuffle=True)
    
    print(f"📊 Training Queue: {len(train_loader)} batches per epoch.")

    for epoch in range(start_epoch, total_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        total_loss = 0
        for step, batch in enumerate(loop):
            optimizer.zero_grad()
            try:
                # Handle dictionary input from DataLoader vs synthetic tensor
                if isinstance(batch, dict):
                    batch_inputs = batch['input_ids']
                else:
                    batch_inputs = batch
                
                loss = trainer.train_step(batch_inputs)
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                loss_val = loss.item()
                total_loss += loss_val
                loop.set_postfix(loss=f"{loss_val:.4f}", avg=f"{total_loss/(step+1):.4f}")
                
                # Checkpoint persistence (Every 20 steps)
                if step % 20 == 0:
                    save_checkpoint(epoch, step, optimizer, trainer.student)
            except Exception as e:
                print(f"\n🛑 Step {step} Failed: {e}")
                return # Strict failure
        
        # Save end of epoch
        save_checkpoint(epoch + 1, 0, optimizer, trainer.student)
        print(f"💾 Epoch {epoch+1} finished and synced to Drive.")

if __name__ == "__main__":
    run_distillation()
