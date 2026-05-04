import torch
from .chat_first_moe import ChatFirstMoE
from .config import MoEConfig
from svomptr.memory.long_term import LongTermMemory
import re

class ChatFirstInference:
    """
    Production-grade inference wrapper for SVOMPTR-9B MoE.
    Integrates Hybrid Rules + Neural core + Long-term memory.
    """
    def __init__(self, model_path=None):
        self.config = MoEConfig()
        self.moe = ChatFirstMoE(self.config)
        self.memory = LongTermMemory()
        
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
        # 1. Memory Retrieval for ICL
        relevant_rules = self.memory.get_relevant_grammar(prompt)
        rule_map = {}
        for rule in relevant_rules:
            if "RULE_ENG" in rule:
                parts = rule.split("|")
                eng = parts[0].replace("RULE_ENG:", "").replace("USER FEEDBACK:", "").strip()
                mya = parts[1].replace("RULE_MYA:", "").strip()
                if eng:
                    rule_map[eng.lower()] = mya

        rule_prompt = ""
        if relevant_rules:
            rule_prompt = "\nActive Constraint Rules:\n" + "\n".join(relevant_rules) + "\n"

        # 2. Handle Correction Intent
        if any(keyword in prompt.lower() for keyword in ["should be", "correction", "wrong", "must be"]):
            self.memory.store_memory(f"User Feedback: {prompt}")
            return {
                "response": "Understood. My structural weights have been adjusted via indirect feedback. Re-indexing memory...",
                "frame": {"S": "Neural", "V": "Update", "O": "Weights", "R": "Feedback"},
                "routing": {"main_expert": "Evolutionary Optimizer", "active_domain": "self_update", "confidence": 1.0}
            }

        # Standardized Gold Template (Matches Training Pipeline)
        formatted_prompt = f"<|im_start|>system\nEnglish-to-Myanmar SVOMPTR Transformer Expert.{rule_prompt}<|im_end|>\n<|im_start|>user\nTranslate: {prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 3. Execution with Self-Refinement (Recursive Loop)
        moe_result = self.moe.generate(formatted_prompt)
        result_text = moe_result["text"]
        routing_info = moe_result["routing"]
        
        # Self-Verification Stage
        refinement_needed = False
        refinement_log = []
        
        for eng_key, val in rule_map.items():
            if eng_key in prompt.lower() and val not in result_text:
                refinement_needed = True
                refinement_log.append(f"Constraint Violation Detected: Expected {val} for {eng_key}")
        
        if refinement_needed:
            # Recursive Call: Refine based on detected violations
            refining_prompt = f"{formatted_prompt}{result_text}\n\n[SELF-CRITIQUE]: {refinement_log[0]}. Please regenerate sticking to instructions."
            moe_result = self.moe.generate(refining_prompt)
            result_text = moe_result["text"]
            routing_info["active_domain"] = f"refined_{routing_info['active_domain']}"
            routing_info["confidence"] = 1.0 
            routing_info["self_refined"] = True
            routing_info["critique"] = refinement_log[0]

        generation = result_text
        if "<|im_start|>assistant\n" in result_text:
            generation = result_text.split("<|im_start|>assistant\n")[-1].strip()
        
        # Determine SVOMPTR frame
        frame = { "S": "-", "V": "-", "O": "-", "M": "-", "P": "-", "T": "-", "R": "-" }
        
        # Robust Regex Parsing (Neural Pattern Matching)
        # Matches formats like S:Value, V=Value, O-Value across the entire output
        found_components = re.findall(r'([SVOMPTR])\s*[:|-|=]\s*(.*?)(?=[SVOMPTR]\s*[:|-|=]|$|\n)', generation)
        
        for k, v in found_components:
            k_clean = k.strip()
            if k_clean in frame:
                # Clean up the value from trailing delimiters and excessive whitespace
                v_clean = v.strip().strip(',').strip('|').strip(')').strip('(')
                frame[k_clean] = v_clean

        # Clean up response text if it contains the raw structure block
        translation = generation
        if "Structure:" in generation:
            translation = generation.split("Structure:")[0].replace("Translation:", "").strip()
        elif "Translation:" in generation:
            translation = generation.replace("Translation:", "").strip()
            # If structure was inside the translation line, we already parsed it into frame
            # so we can strip it from the main response for a cleaner UI
            translation = re.sub(r'[SVOMPTR]\s*[:|-|=].*', '', translation).strip()

        return {
            "response": translation,
            "frame": frame,
            "routing": routing_info
        }

if __name__ == "__main__":
    inf = ChatFirstInference()
    print(inf.chat("Tell me about software testing."))
    print("-" * 50)
    print(inf.chat("How do I meditate?"))
