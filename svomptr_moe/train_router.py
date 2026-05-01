# SVOMPTR-9B MoE Training Suite
# Part 3: Router Training

import os
import torch
import torch.nn as nn
import torch.optim as optim

def train_router():
    print("Task: Training Gate/Router Weights")
    print("Dataset: Domain classification pairs (text, label)")
    print("Loss: CrossEntropyLoss + MoE Balance Loss (Expert Load Balancing)")
    
    brain_dir = os.environ.get("SVOMPTR_BRAIN_PATH", "/content/drive/MyDrive/svomptr_brain")
    if os.path.exists(brain_dir):
        print(f"SVOMPTR Brain detected at {brain_dir}")
        router_dir = os.path.join(brain_dir, "weights", "domain_router_final")
        os.makedirs(router_dir, exist_ok=True)
    else:
        print("Running in local mode. SVOMPTR Brain not detected.")
        router_dir = "./svomptr_export/domain_router_final"
        os.makedirs(router_dir, exist_ok=True)
        
    try:
        # Avoid circular import by dynamic importing the model
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from domain_router import DomainRouter
        
        # Setup dummy domains for indexing
        domains = ["chat", "software", "medicine", "engineering", "buddhism", "history", "science"]
        num_sub_experts = len(domains) - 1
        
        print("Initializing DomainRouter (384 -> classifier)...")
        router = DomainRouter(384, num_sub_experts)
        
        # Create Synthetic Minimal Dataset for Routing Training
        # In actual usage, this should read from brain_dir/datasets/routing_data.jsonl
        train_data = []
        for i, d in enumerate(domains[1:]): # 0 to 5 indexes
            train_data.append((f"Tell me about {d} concepts and technical terms.", i))
            train_data.append((f"How does {d} work at a deep level?", i))
            
        print(f"Mocked {len(train_data)} training pairs for routing. Precomputing embeddings...")
        
        if router.encoder is None:
            print("SentenceTransformer not installed! Router cannot train. Skipping.")
            return

        X_emb = router.encoder.encode([item[0] for item in train_data])
        X = torch.tensor(X_emb, dtype=torch.float32)
        y = torch.tensor([item[1] for item in train_data], dtype=torch.long)
        
        optimizer = optim.AdamW(router.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        epochs = 50
        print("Starting training loop...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = router.classifier(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.save_step = optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                acc = (logits.argmax(1) == y).float().mean()
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Acc: {acc:.2f}")

        save_path = os.path.join(router_dir, "router_weights.pth")
        torch.save(router.state_dict(), save_path)
        print(f"Router training completed successfully. Weights saved to {save_path}")

    except ImportError as e:
        print(f"Dependencies missing for Router Training: {e}")
        
if __name__ == "__main__":
    train_router()
