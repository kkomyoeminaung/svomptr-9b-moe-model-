import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainRouter(nn.Module):
    """
    Lightweight classifier that detects domain labels from query embeddings.
    """
    def __init__(self, input_dim, num_sub_experts):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384 # all-MiniLM-L6-v2 dimension
        except ImportError:
            self.encoder = None
            self.embedding_dim = input_dim
            
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_sub_experts)
        )
        
        import os
        brain_dir = os.environ.get("SVOMPTR_BRAIN_PATH", "/content/drive/MyDrive/svomptr_brain")
        router_weights = os.path.join(brain_dir, "weights", "domain_router_final", "router_weights.pth")
        if os.path.exists(router_weights):
            try:
                self.load_state_dict(torch.load(router_weights, map_location="cpu"))
                print(f"[MoE Router] Loaded semantic classifier weights from {router_weights}")
            except Exception as e:
                print(f"[MoE Router] Failed to load custom weights: {e}")
        
    def get_route(self, query_text):
        """
        Semantic routing using sentence-transformers and trainable classifier.
        Returns: (expert_id_0_to_11, confidence_0_to_1)
        """
        if self.encoder is None:
            # Fallback to dummy implementation if sentence-transformers is missing
            return 0, 0.2
            
        with torch.no_grad():
            emb = self.encoder.encode([query_text])
            emb_tensor = torch.tensor(emb, dtype=torch.float32)
            logits = self.classifier(emb_tensor)
            probs = torch.softmax(logits, dim=-1)
            domain_idx = probs.argmax().item()
            confidence = probs.max().item()
            
        return domain_idx, confidence
