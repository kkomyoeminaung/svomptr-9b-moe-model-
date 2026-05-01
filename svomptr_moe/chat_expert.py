import torch.nn as nn

class ChatExpert(nn.Module):
    """
    Expert 0 (2.5B): The primary conversational backbone.
    Based on Gemma-2B/Qwen-2.5-1.5B scales with SVOMPTR-9B alignment.
    """
    def __init__(self, model_path=None, config=None):
        super().__init__()
        from .config import MoEConfig
        self.config = config if config else MoEConfig()
        self.name = "chat_expert_core"
        self.hidden_dim = self.config.hidden_dim
        self.projector = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        import os
        if model_path is None:
            brain_dir = os.environ.get("SVOMPTR_BRAIN_PATH", "/content/drive/MyDrive/svomptr_brain")
            brain_model_path = os.path.join(brain_dir, "weights", "chat_expert_final")
            if os.path.exists(brain_model_path):
                model_path = brain_model_path
            else:
                model_path = "Qwen/Qwen2.5-1.5B-Instruct"

        try:
            from transformers import pipeline
            self.pipe = pipeline("text-generation", 
                                 model=model_path, 
                                 torch_dtype="auto", 
                                 device_map="auto")
            print(f"[MoE] Chat Expert loaded: {model_path}")
        except Exception as e:
            print(f"[MoE] Chat Expert init failed (missing transformers?): {e}")
            self.pipe = None

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
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                return result[0]['generated_text']
            except Exception as e:
                return f"[MoE Generation Error]: {str(e)}"
                
        # Simulate generation for demo/arch purposes
        return f"Translation: {query} (Simulated Burmese Translation)\nStructure: S: Model, V: Analyzes, O: {query}"
