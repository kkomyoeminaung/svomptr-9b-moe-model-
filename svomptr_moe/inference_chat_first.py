
import torch
from svomptr_9b.svomptr.core.svomptr_rules import SVOMPTRRuleEngine
from svomptr_9b.svomptr.memory.long_term import LongTermMemory

class ChatFirstInference:
    def __init__(self):
        self.engine = SVOMPTRRuleEngine()
        self.memory = LongTermMemory()
        
    def infer(self, text):
        # Hybrid Logic: Rule Extraction + Memory Retrieval
        rules = self.engine.parse(text)
        memories = self.memory.get_relevant_context(text)
        
        response = f"Primary analysis: S={rules.S}, V={rules.V}. Context found."
        return {
            "text": response,
            "structure": rules.__dict__,
            "memories": memories
        }

    def chat(self, message):
        """Main entry point for chat interaction."""
        # 1. Self-Reflection: Match against rules
        rules = self.engine.parse(message)
        
        # 2. Retrieve long-term memory/rules
        context = self.memory.get_relevant_grammar(message)
        
        # 3. Formulate response (Hybrid: Template + Logic)
        response = f"I've analyzed your input using the SVOMPTR neural map. Structure detected: {rules.S} -> {rules.V}."
        
        if context:
            response += f"\nNote: I found {len(context)} relevant grammar rules in my memory."

        # Clean frame for JSON serialization
        frame_dict = {k: v for k, v in rules.__dict__.items() if k not in ['tokens', 'raw_text']}
        if 'sentence_type' in frame_dict and hasattr(frame_dict['sentence_type'], 'value'):
            frame_dict['sentence_type'] = frame_dict['sentence_type'].value

        return {
            "response": response,
            "frame": frame_dict
        }
