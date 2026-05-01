import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    def train_epoch(self):
        raise NotImplementedError("Each phase trainer must implement train_epoch")

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                # Basic validation logic
                pass
        self.model.train()

    def save_checkpoint(self, path):
        import os
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config
            }
            torch.save(checkpoint, path)
            
            # Critical verification
            if os.path.exists(path):
                size = os.path.getsize(path)
                if size > 0:
                    print(f"✅ Checkpoint verified and saved to {path} ({size/1024/1024:.2f} MB)")
                else:
                    print(f"❌ ERROR: Saved file {path} is empty (0 bytes)!")
            else:
                print(f"❌ ERROR: File {path} was not created after torch.save!")
        except Exception as e:
            print(f"❌ CRITICAL ERROR saving checkpoint: {e}")
