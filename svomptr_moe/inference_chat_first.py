import torch
from .chat_first_moe import ChatFirstMoE
from .config import MoEConfig
import re

class ChatFirstInference:
    """
    Production-grade inference wrapper for SVOMPTR-9B MoE.
    Runs on CPU by defaulting always-on Chat expert (2.5B).
    """
    def __init__(self, model_path=None):
        self.config = MoEConfig()
        self.moe = ChatFirstMoE(self.config)
        
        if model_path:
            try:
                self.moe.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"[MoE] Loaded weights from {model_path}")
            except Exception as e:
                print(f"[MoE] Failed to load weights from {model_path}: {e}")
                print("[MoE] Falling back to initialized weights")
        else:
            print("[MoE] No model_path provided. Running with uninitialized / base mock weights.")

    def chat(self, prompt: str):
        formatted_prompt = f"<|im_start|>user\nTranslate and analyze: {prompt}<|im_end|>\n<|im_start|>model\n"
        
        # Execute MoE logic
        result = self.moe.generate(formatted_prompt)
        
        # Extract everything after the prompt to prevent echoing
        generation = result
        if "<|im_start|>model\n" in result:
            generation = result.split("<|im_start|>model\n")[-1].strip()
        
        # Determine SVOMPTR frame
        frame = { "S": "-", "V": "-", "O": "-", "M": "-", "P": "-", "T": "-", "R": "-" }
        
        # Parse output mapping "Translation:" and "Structure: S:..., V:..., O:..."
        translation = generation
        
        if "Structure:" in generation:
            parts = generation.split("Structure:", 1)
            translation = parts[0].replace("Translation:", "").strip()
            structure_str = parts[1].strip()
            
            # Simple extractor for S, V, O etc.
            pairs = [p.strip() for p in structure_str.split(",")]
            for p in pairs:
                if ":" in p:
                    k, v = p.split(":", 1)
                    k = k.strip()
                    if k in frame:
                        frame[k] = v.strip()
                        
        elif "Translation:" in generation:
            translation = generation.replace("Translation:", "").strip()

        return {
            "response": translation,
            "frame": frame
        }

if __name__ == "__main__":
    inf = ChatFirstInference()
    print(inf.chat("Tell me about software testing."))
    print("-" * 50)
    print(inf.chat("How do I meditate?"))
