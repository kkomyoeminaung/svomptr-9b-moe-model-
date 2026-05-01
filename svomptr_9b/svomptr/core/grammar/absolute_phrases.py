# /svomptr_9b/svomptr/core/grammar/absolute_phrases.py

"""
Absolute Phrases Handler
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AbsolutePhraseInfo:
    is_absolute: bool
    phrase: str
    noun: str
    participle: str
    position: int  # 0 = beginning, 1 = middle, 2 = end


class AbsolutePhraseHandler:
    """Handles absolute phrases (nominative absolutes)"""
    
    def __init__(self):
        pass
    
    def detect_absolute_phrase(self, tokens: List[str]) -> Optional[AbsolutePhraseInfo]:
        """
        Detect absolute phrase
        Example: "Weather permitting, we will go"
        """
        sentence = " ".join(tokens)
        
        # Pattern 1: Noun + participle at beginning (ends with comma)
        if "," in sentence:
            # Find phrase before comma
            comma_pos = sentence.find(",")
            phrase = sentence[:comma_pos]
            
            phrase_words = phrase.split()
            if len(phrase_words) >= 2:
                noun = phrase_words[0]
                participle = phrase_words[1]
                
                # Check if second word looks like a participle
                if participle.endswith("ing") or participle.endswith("ed"):
                    return AbsolutePhraseInfo(
                        is_absolute=True,
                        phrase=phrase,
                        noun=noun,
                        participle=participle,
                        position=0  # beginning
                    )
        
        # Pattern 2: With/without + noun + participle
        words = [t.lower() for t in tokens]
        for i, word in enumerate(words):
            if word in ["with", "without"]:
                if i + 2 < len(words):
                    noun = tokens[i + 1]
                    participle = tokens[i + 2]
                    if participle.endswith("ing") or participle.endswith("ed"):
                        phrase = " ".join(tokens[i:i+3])
                        return AbsolutePhraseInfo(
                            is_absolute=True,
                            phrase=phrase,
                            noun=noun,
                            participle=participle,
                            position=0 if i == 0 else 1
                        )
        
        return None
    
    def get_absolute_phrase_rules(self) -> Dict:
        """Get absolute phrase usage rules"""
        return {
            "structure": {
                "description": "Noun + Participle (သို့) Noun + Being + Complement",
                "example": "Weather permitting / His work done / All things considered"
            },
            "punctuation": {
                "rule": "Absolute phrase နောက်တွင် comma (,) ခံရမည်",
                "example": "Weather permitting, we will go."
            }
        }
