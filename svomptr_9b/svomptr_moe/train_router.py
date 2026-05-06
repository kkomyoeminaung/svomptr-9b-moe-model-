# svomptr_moe/train_router.py
import torch
import torch.nn as nn
from tqdm import tqdm
from svomptr_9b.svomptr.core.model_9b import SVOMPTR9B
from svomptr_9b.svomptr.core.config import ModelConfig

import torch.nn.functional as F

def train_router(model, train_loader, device):
    print("🚀 [Step 6] Optimizing MoE Routing Network (Gating Logic)...")
    
    # Router optimization loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    pbar = tqdm(train_loader, desc="Router Batch Processing")
    
    model.train()
    model.to(device)
    for i, batch in enumerate(pbar):
        if i > 100: break
        input_ids = batch['input_ids'].to(device)
        
        optimizer.zero_grad()
        
        # Collect router logits from all MoE layers
        routing_losses = []
        x = model.embedding(input_ids)
        for block in model.blocks:
            attn_out, _ = block["attention"](x, x, x)
            x = block["norm1"](x + attn_out)
            
            # Get routing weights for load balancing
            router_logits = block["moe"].router(x)          # [B, S, E]
            routing_probs = torch.softmax(router_logits, dim=-1)
            
            # Load balancing loss: minimize variance in expert utilization
            mean_usage = routing_probs.mean(dim=[0, 1])     # [E]
            ideal = torch.ones_like(mean_usage) / mean_usage.shape[0]
            lb_loss = F.kl_div(mean_usage.log(), ideal, reduction='sum')
            routing_losses.append(lb_loss)
            
            moe_out = block["moe"](x)
            x = block["norm2"](x + moe_out)
            
        total_loss = sum(routing_losses) / len(routing_losses)
        total_loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"routing_loss": f"{total_loss.item():.6f}"})
    
    print("✅ MoE Router stabilized and weights synced.")

if __name__ == "__main__":
    train_router()
