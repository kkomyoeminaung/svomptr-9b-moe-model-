from sentence_transformers import SentenceTransformer
import torch.nn as nn

class DomainRouter:
    def __init__(self, embed_dim=384, num_classes=10):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def route(self, text):
        embedding = self.encoder.encode(text)
        return self.classifier(torch.tensor(embedding))

    def get_route(self, text):
        import torch
        logits = self.route(text)
        probs = torch.softmax(logits, dim=-1)
        confidence, idx = torch.max(probs, dim=-1)
        return idx.item(), confidence.item()
