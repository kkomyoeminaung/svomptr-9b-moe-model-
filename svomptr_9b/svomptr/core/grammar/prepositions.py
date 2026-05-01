# /svomptr_9b/svomptr/core/grammar/prepositions.py

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class PrepositionType(Enum):
    TIME = "time"          # at, on, in, during, for, since
    PLACE = "place"        # at, on, in, under, above, below, between
    DIRECTION = "direction" # to, from, into, onto, through, across
    MANNER = "manner"      # by, with, like, without
    PURPOSE = "purpose"    # for, to
    CAUSE = "cause"        # because of, due to, owing to
    AGENT = "agent"        # by (passive agent)
    POSSESSION = "possession" # of, with
    MEASURE = "measure"    # by, at (price/speed)


@dataclass
class PrepositionInfo:
    preposition: str
    type: PrepositionType
    object: str
    position: int  # Position in sentence
    meaning: str


class PrepositionHandler:
    """
    Handles all prepositions and prepositional phrases
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Comprehensive preposition lists by type
        self.prepositions = {
            PrepositionType.TIME: {
                "words": ["at", "on", "in", "during", "for", "since", "by", "until", "till", "throughout", "within"],
                "meanings": {
                    "at": "တွင် (အချိန်အတိအကျ)",
                    "on": "တွင် (နေ့ရက်)",
                    "in": "တွင် (လ/နှစ်/ရာသီ)",
                    "during": "အတွင်း",
                    "for": "ကြာ",
                    "since": "ကတည်းက",
                    "by": "အတွင်း (သတ်မှတ်ချိန်)",
                    "until": "အထိ"
                }
            },
            PrepositionType.PLACE: {
                "words": ["at", "on", "in", "inside", "outside", "under", "above", "below", "between", "among", 
                         "behind", "in front of", "next to", "near", "far from", "opposite", "by", "beside"],
                "meanings": {
                    "at": "မှာ (နေရာအတိအကျ)",
                    "on": "ပေါ်မှာ",
                    "in": "ထဲမှာ",
                    "under": "အောက်မှာ",
                    "above": "အပေါ်မှာ",
                    "between": "ကြား",
                    "behind": "နောက်မှာ"
                }
            },
            PrepositionType.DIRECTION: {
                "words": ["to", "from", "into", "onto", "toward", "towards", "through", "across", "along", 
                         "around", "past", "via", "up", "down", "off", "out of", "away from"],
                "meanings": {
                    "to": "သို့",
                    "from": "မှ",
                    "into": "ထဲသို့",
                    "onto": "ပေါ်သို့",
                    "toward": "ဆီသို့",
                    "through": "ဖြတ်၍",
                    "across": "ဖြတ်၍",
                    "up": "အပေါ်သို့",
                    "down": "အောက်သို့"
                }
            },
            PrepositionType.MANNER: {
                "words": ["by", "with", "like", "without", "in a ... manner", "in a ... way"],
                "meanings": {
                    "by": "ဖြင့်",
                    "with": "နှင့်",
                    "like": "ကဲ့သို့",
                    "without": "မပါဘဲ"
                }
            },
            PrepositionType.PURPOSE: {
                "words": ["for", "to"],
                "meanings": {
                    "for": "အတွက်",
                    "to": "ရန်"
                }
            },
            PrepositionType.CAUSE: {
                "words": ["because of", "due to", "owing to", "on account of", "thanks to"],
                "meanings": {
                    "because of": "ကြောင့်",
                    "due to": "ကြောင့်",
                    "owing to": "ကြောင့်"
                }
            },
            PrepositionType.AGENT: {
                "words": ["by"],
                "meanings": {
                    "by": "မှ (passive voice)"
                }
            },
            PrepositionType.POSSESSION: {
                "words": ["of", "with"],
                "meanings": {
                    "of": "၏",
                    "with": "ပါသော"
                }
            }
        }
        
        # Myanmar prepositions
        self.myanmar_prepositions = {
            "မှာ": PrepositionType.PLACE,
            "ထဲမှာ": PrepositionType.PLACE,
            "ပေါ်မှာ": PrepositionType.PLACE,
            "အောက်မှာ": PrepositionType.PLACE,
            "ဆီ": PrepositionType.DIRECTION,
            "ဘက်": PrepositionType.DIRECTION,
            "အတွက်": PrepositionType.PURPOSE,
            "ကြောင့်": PrepositionType.CAUSE,
            "နှင့်": PrepositionType.MANNER,
            "ဖြင့်": PrepositionType.MANNER
        }
    
    def detect_prepositions(self, tokens: List[str]) -> List[PrepositionInfo]:
        """
        Detect all prepositions in sentence
        """
        prepositions_found = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            # Check Myanmar prepositions first
            if token_lower in self.myanmar_prepositions:
                prep_type = self.myanmar_prepositions[token_lower]
                # Find object (next token)
                obj = tokens[i + 1] if i + 1 < len(tokens) else ""
                prepositions_found.append(PrepositionInfo(
                    preposition=token,
                    type=prep_type,
                    object=obj,
                    position=i,
                    meaning=self._get_meaning(token)
                ))
                continue
            
            # Check English prepositions
            for prep_type, info in self.prepositions.items():
                if token_lower in info["words"]:
                    # Find object (what follows the preposition)
                    obj = tokens[i + 1] if i + 1 < len(tokens) else ""
                    prepositions_found.append(PrepositionInfo(
                        preposition=token,
                        type=prep_type,
                        object=obj,
                        position=i,
                        meaning=self._get_meaning(token)
                    ))
                    break
        
        return prepositions_found
    
    def _get_meaning(self, preposition: str) -> str:
        """Get Myanmar meaning of preposition"""
        for prep_type, info in self.prepositions.items():
            if preposition in info["meanings"]:
                return info["meanings"][preposition]
        
        # Check Myanmar prepositions
        if preposition in self.myanmar_prepositions:
            return self._get_myanmar_preposition_meaning(preposition)
        
        return ""
    
    def _get_myanmar_preposition_meaning(self, prep: str) -> str:
        """Get meaning of Myanmar preposition"""
        meanings = {
            "မှာ": "မှာ (နေရာ)",
            "ထဲမှာ": "ထဲတွင်",
            "ပေါ်မှာ": "အပေါ်တွင်",
            "အောက်မှာ": "အောက်တွင်",
            "ဆီ": "သို့",
            "အတွက်": "အတွက်",
            "ကြောင့်": "သောကြောင့်",
            "နှင့်": "နှင့်အတူ",
            "ဖြင့်": "ဖြင့်"
        }
        return meanings.get(prep, "")
    
    def get_prepositional_phrase(self, tokens: List[str], prep_info: PrepositionInfo) -> str:
        """Extract the full prepositional phrase"""
        # From preposition to end of PP
        # PP can be: preposition + noun phrase (usually 1-3 words)
        start = prep_info.position
        end = min(start + 3, len(tokens))  # Simplified: PP up to 3 words
        return " ".join(tokens[start:end])
