# /svomptr_9b/svomptr/core/grammar/tense.py

from enum import Enum
from typing import Dict, List, Optional, Tuple
import re


class Tense(Enum):
    """12 English Tenses"""
    PRESENT_SIMPLE = "present_simple"
    PRESENT_CONTINUOUS = "present_continuous"
    PRESENT_PERFECT = "present_perfect"
    PRESENT_PERFECT_CONTINUOUS = "present_perfect_continuous"
    PAST_SIMPLE = "past_simple"
    PAST_CONTINUOUS = "past_continuous"
    PAST_PERFECT = "past_perfect"
    PAST_PERFECT_CONTINUOUS = "past_perfect_continuous"
    FUTURE_SIMPLE = "future_simple"
    FUTURE_CONTINUOUS = "future_continuous"
    FUTURE_PERFECT = "future_perfect"
    FUTURE_PERFECT_CONTINUOUS = "future_perfect_continuous"


class TenseHandler:
    """
    Handles all 12 tenses
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Time markers for tense detection
        self.time_markers = {
            Tense.PRESENT_SIMPLE: ["always", "usually", "often", "sometimes", "never", "every day"],
            Tense.PRESENT_CONTINUOUS: ["now", "right now", "at the moment", "currently"],
            Tense.PRESENT_PERFECT: ["already", "yet", "just", "ever", "never", "so far"],
            Tense.PRESENT_PERFECT_CONTINUOUS: ["for", "since", "lately", "recently"],
            Tense.PAST_SIMPLE: ["yesterday", "last", "ago", "in 2020", "previously"],
            Tense.PAST_CONTINUOUS: ["while", "when", "at that moment"],
            Tense.PAST_PERFECT: ["already", "by the time", "before", "after"],
            Tense.PAST_PERFECT_CONTINUOUS: ["for", "since", "before"],
            Tense.FUTURE_SIMPLE: ["tomorrow", "next", "soon", "later"],
            Tense.FUTURE_CONTINUOUS: ["at this time tomorrow", "at that time"],
            Tense.FUTURE_PERFECT: ["by then", "by the time"],
            Tense.FUTURE_PERFECT_CONTINUOUS: ["for", "by the time"]
        }
        
        # Myanmar time markers
        self.myanmar_time_markers = {
            "ဒီနေ့": Tense.PRESENT_SIMPLE,
            "အခု": Tense.PRESENT_CONTINUOUS,
            "မနေ့က": Tense.PAST_SIMPLE,
            "မနက်ဖြန်": Tense.FUTURE_SIMPLE,
            "ပြီးပြီ": Tense.PRESENT_PERFECT,
            "နေ့တိုင်း": Tense.PRESENT_SIMPLE,
            "အမြဲ": Tense.PRESENT_SIMPLE,
        }
        
        # Auxiliary verb patterns
        self.aux_patterns = {
            Tense.PRESENT_SIMPLE: {"do": 1, "does": 1, "base": 1},
            Tense.PRESENT_CONTINUOUS: {"am": 1, "is": 1, "are": 1, "ing": 1},
            Tense.PRESENT_PERFECT: {"have": 1, "has": 1, "past_participle": 1},
            Tense.PRESENT_PERFECT_CONTINUOUS: {"have": 1, "has": 1, "been": 1, "ing": 1},
            Tense.PAST_SIMPLE: {"ed": 1, "irregular": 1},
            Tense.PAST_CONTINUOUS: {"was": 1, "were": 1, "ing": 1},
            Tense.PAST_PERFECT: {"had": 1, "past_participle": 1},
            Tense.PAST_PERFECT_CONTINUOUS: {"had": 1, "been": 1, "ing": 1},
            Tense.FUTURE_SIMPLE: {"will": 1, "base": 1},
            Tense.FUTURE_CONTINUOUS: {"will": 1, "be": 1, "ing": 1},
            Tense.FUTURE_PERFECT: {"will": 1, "have": 1, "past_participle": 1},
            Tense.FUTURE_PERFECT_CONTINUOUS: {"will": 1, "have": 1, "been": 1, "ing": 1},
        }
    
    def detect_tense(self, tokens: List[str]) -> Tuple[Tense, float]:
        """
        Detect tense from tokens
        Returns: (tense, confidence)
        """
        sentence = " ".join(tokens).lower()
        
        # Check time markers first (highest confidence)
        for markers, tense in self.time_markers.items():
            for tm in markers:
                if tm in sentence:
                    return tense, 0.95
        
        # Check Myanmar time markers
        for marker, tense in self.myanmar_time_markers.items():
            if marker in sentence:
                return tense, 0.95
        
        # Check auxiliary verb patterns
        best_tense = Tense.PRESENT_SIMPLE
        best_score = 0
        
        for tense, pattern in self.aux_patterns.items():
            score = 0
            for aux, weight in pattern.items():
                if aux == "ing" and any(w.endswith("ing") for w in tokens):
                    score += weight
                elif aux == "past_participle":
                    # Check for common past participles
                    if any(w.endswith("ed") or w in ["gone", "done", "seen", "eaten"] for w in tokens):
                        score += weight
                elif any(w == aux for w in tokens):
                    score += weight
            
            if score > best_score:
                best_score = score
                best_tense = tense
        
        return best_tense, min(0.8, best_score / 4)
    
    def get_tense_marker(self, tense: Tense) -> str:
        """Get the auxiliary verb marker for a tense"""
        markers = {
            Tense.PRESENT_SIMPLE: "",
            Tense.PRESENT_CONTINUOUS: "be + V-ing",
            Tense.PRESENT_PERFECT: "have/has + V-ed/Ven",
            Tense.PRESENT_PERFECT_CONTINUOUS: "have/has + been + V-ing",
            Tense.PAST_SIMPLE: "V-ed/V2",
            Tense.PAST_CONTINUOUS: "was/were + V-ing",
            Tense.PAST_PERFECT: "had + V-ed/Ven",
            Tense.PAST_PERFECT_CONTINUOUS: "had + been + V-ing",
            Tense.FUTURE_SIMPLE: "will + V",
            Tense.FUTURE_CONTINUOUS: "will + be + V-ing",
            Tense.FUTURE_PERFECT: "will + have + V-ed/Ven",
            Tense.FUTURE_PERFECT_CONTINUOUS: "will + have + been + V-ing",
        }
        return markers.get(tense, "")
    
    def tense_to_myanmar(self, tense: Tense) -> str:
        """Convert tense to Myanmar explanation"""
        myanmar_names = {
            Tense.PRESENT_SIMPLE: "ပစ္စုပ္ပန်သာမန်",
            Tense.PRESENT_CONTINUOUS: "ပစ္စုပ္ပန်ကြာရှည်",
            Tense.PRESENT_PERFECT: "ပစ္စုပ္ပန်ပြီးစီး",
            Tense.PRESENT_PERFECT_CONTINUOUS: "ပစ္စုပ္ပန်ပြီးစီးကြာရှည်",
            Tense.PAST_SIMPLE: "အတိတ်သာမန်",
            Tense.PAST_CONTINUOUS: "အတိတ်ကြာရှည်",
            Tense.PAST_PERFECT: "အတိတ်ပြီးစီး",
            Tense.PAST_PERFECT_CONTINUOUS: "အတိတ်ပြီးစီးကြာရှည်",
            Tense.FUTURE_SIMPLE: "အနာဂတ်သာမန်",
            Tense.FUTURE_CONTINUOUS: "အနာဂတ်ကြာရှည်",
            Tense.FUTURE_PERFECT: "အနာဂတ်ပြီးစီး",
            Tense.FUTURE_PERFECT_CONTINUOUS: "အနာဂတ်ပြီးစီးကြာရှည်",
        }
        return myanmar_names.get(tense, "အချိန်သရုပ်")
