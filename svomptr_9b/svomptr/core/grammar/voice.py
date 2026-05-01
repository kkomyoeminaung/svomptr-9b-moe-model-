# /svomptr_9b/svomptr/core/grammar/voice.py

from enum import Enum
from typing import Dict, List, Optional, Tuple


class Voice(Enum):
    ACTIVE = "active"
    PASSIVE = "passive"


class VoiceHandler:
    """
    Handles Active/Passive voice transformation
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Passive auxiliary verbs
        self.passive_auxiliaries = {
            "present": ["am", "is", "are"],
            "past": ["was", "were"],
            "perfect": ["been"],
            "continuous": ["being"]
        }
        
        # Past participle endings (အတိတ်ကာလပြုပုံ)
        self.past_participle_endings = ["ed", "en", "t", "ne", "d"]
        
        # Irregular past participles
        self.irregular_past_participle = {
            "eat": "eaten", "go": "gone", "see": "seen", "do": "done",
            "write": "written", "speak": "spoken", "break": "broken",
            "choose": "chosen", "drive": "driven", "forget": "forgotten",
            "give": "given", "hide": "hidden", "ride": "ridden",
            "rise": "risen", "shake": "shaken", "take": "taken"
        }
    
    def detect_passive(self, tokens: List[str]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect passive voice
        Returns: (is_passive, original_object, agent)
        
        Example:
          "The mouse was eaten by the cat"
          → is_passive = True, original_object = "The mouse", agent = "the cat"
        """
        if len(tokens) < 3:
            return False, None, None
        
        # Look for passive pattern: [noun] + [be verb] + [past participle] + (by + noun)
        # Find the verb (past participle)
        past_participle_index = -1
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            # Check if it's a past participle
            if self._is_past_participle(token_lower):
                past_participle_index = i
                break
        
        if past_participle_index == -1:
            return False, None, None
        
        # Check if there's a be verb before it
        be_verb_index = -1
        for i in range(past_participle_index - 1, -1, -1):
            if tokens[i].lower() in self.passive_auxiliaries["present"] + \
               self.passive_auxiliaries["past"]:
                be_verb_index = i
                break
        
        if be_verb_index == -1:
            return False, None, None
        
        # The subject before be verb is actually the original object
        original_object = " ".join(tokens[:be_verb_index]) if be_verb_index > 0 else None
        
        # Look for "by" phrase (agent)
        agent = None
        for i in range(past_participle_index + 1, len(tokens)):
            if tokens[i].lower() == "by" and i + 1 < len(tokens):
                agent = " ".join(tokens[i + 1:])
                break
        
        return True, original_object, agent
    
    def _is_past_participle(self, word: str) -> bool:
        """Check if word is a past participle form"""
        # Check irregular verbs
        if word in self.irregular_past_participle.values():
            return True
        # Check regular -ed ending
        if word.endswith("ed"):
            return True
        # Check other common endings
        for ending in self.past_participle_endings:
            if word.endswith(ending) and len(word) > 3:
                return True
        return False
    
    def transform_to_active(self, passive_sentence: str, original_object: str, agent: str) -> str:
        """
        Transform passive to active
        Example: "The mouse was eaten by the cat" → "The cat ate the mouse"
        """
        if not agent:
            return passive_sentence
        return f"{agent} ate {original_object}"
    
    def transform_to_passive(self, active_sentence: str, tense: str = "past") -> str:
        """
        Transform active to passive (simplified)
        Example: "The cat ate the mouse" → "The mouse was eaten by the cat"
        """
        # Simplified implementation - full version would do proper parsing
        parts = active_sentence.split()
        if len(parts) < 3:
            return active_sentence
        
        # Assume pattern: [subject] [verb] [object]
        subject = parts[0]
        verb = parts[1]
        object_part = " ".join(parts[2:])
        
        # Choose be verb
        if tense == "present":
            be_verb = "is"
        else:
            be_verb = "was"
        
        return f"{object_part} {be_verb} {verb}en by {subject}"
