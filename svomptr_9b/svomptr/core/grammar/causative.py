# /svomptr_9b/svomptr/core/grammar/causative.py

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CausativeInfo:
    is_causative: bool
    causative_verb: Optional[str]
    person: Optional[str]  # Who is being caused
    action: Optional[str]   # What action is being caused
    is_passive_causative: bool  # have/get + object + past participle


class CausativeHandler:
    """
    Handles causative verbs (make, let, have, get, help)
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        self.causative_verbs = {
            "make": {
                "pattern": "make + person + verb",
                "meaning": "အတင်းခိုင်းစေသည်",
                "force_level": "high"
            },
            "let": {
                "pattern": "let + person + verb",
                "meaning": "ခွင့်ပြုသည်",
                "force_level": "low"
            },
            "have": {
                "pattern": "have + person + verb (active)",
                "meaning": "ခိုင်းစေသည် (တာဝန်ပေး)",
                "force_level": "medium"
            },
            "get": {
                "pattern": "get + person + to + verb",
                "meaning": "သဘောတူညီစေသည်",
                "force_level": "medium"
            },
            "help": {
                "pattern": "help + person + (to) verb",
                "meaning": "ကူညီသည်",
                "force_level": "low"
            }
        }
        
        # Passive causative pattern: have/get + object + past participle
        # Example: "I had my car repaired"
        self.passive_causative_pattern = ["have", "get"]
    
    def detect_causative(self, tokens: List[str]) -> CausativeInfo:
        """
        Detect causative structure
        """
        if len(tokens) < 3:
            return CausativeInfo(False, None, None, None, False)
        
        # Look for causative verb
        for i, token in enumerate(tokens):
            if token.lower() in self.causative_verbs:
                causative_verb = token.lower()
                
                # Check for passive causative (have/get + object + past participle)
                if causative_verb in self.passive_causative_pattern and i + 2 < len(tokens):
                    # Check if next token might be past participle
                    next_word = tokens[i + 2].lower()
                    if next_word.endswith("ed") or next_word in ["repaired", "fixed", "cleaned", "washed"]:
                        return CausativeInfo(
                            is_causative=True,
                            causative_verb=causative_verb,
                            person=tokens[i + 1] if i + 1 < len(tokens) else None,
                            action=tokens[i + 2],
                            is_passive_causative=True
                        )
                
                # Active causative: verb + person + action
                if i + 2 < len(tokens):
                    return CausativeInfo(
                        is_causative=True,
                        causative_verb=causative_verb,
                        person=tokens[i + 1],
                        action=tokens[i + 2],
                        is_passive_causative=False
                    )
        
        return CausativeInfo(False, None, None, None, False)
    
    def get_meaning(self, causative_verb: str) -> str:
        """Get Myanmar meaning of causative verb"""
        if causative_verb in self.causative_verbs:
            return self.causative_verbs[causative_verb]["meaning"]
        return ""
