import torch.nn as nn

class SubExpert(nn.Module):
    """
    Sub-Experts (0.5B): Domain specialists.
    Lightweight, high-accuracy adapters or mini-models.
    """
    def __init__(self, domain, model_path=None, config=None):
        super().__init__()
        from .config import MoEConfig
        self.config = config if config else MoEConfig()
        self.domain = domain
        self.name = f"expert_{domain}_0.5b"
        self.hidden_dim = self.config.hidden_dim
        self.projector = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pipe = None
        
        import os
        if model_path is None:
            brain_dir = os.environ.get("SVOMPTR_BRAIN_PATH", "/content/drive/MyDrive/svomptr_brain")
            brain_model_path = os.path.join(brain_dir, "weights", "sub_experts", domain)
            if os.path.exists(brain_model_path):
                model_path = brain_model_path

        if model_path:
            try:
                from transformers import pipeline
                self.pipe = pipeline("text-generation", 
                                     model=model_path, 
                                     torch_dtype="auto", 
                                     device_map="auto")
                print(f"[MoE] {domain.capitalize()} Expert loaded: {model_path}")
            except Exception as e:
                print(f"[MoE] {domain.capitalize()} Expert init failed: {e}")
        else:
            print(f"[MoE] {domain.capitalize()} Expert (0.5B) Initialized in mock mode.")

    def forward(self, x, mask=None):
        # Bug #16 fix: Real neural transformation
        return self.projector(x)

    def save_expert(self, path):
        """Save expert specific weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Expert weights saved to {path}")

    def load_expert(self, path):
        """Load expert specific weights."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"Expert weights loaded from {path}")

    def generate(self, query, max_tokens=256):
        if self.pipe:
            try:
                result = self.pipe(
                    query, 
                    max_new_tokens=max_tokens,
                    temperature=0.5,
                    top_p=0.9,
                    do_sample=True
                )
                return result[0]['generated_text']
            except Exception as e:
                return f"[MoE Generation Error]: {str(e)}"
                
        # Simulate domain specialty
        return f"Deep Technical Insights in {self.domain.upper()} regarding '{query}'."
