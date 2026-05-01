# /svomptr_9b/svomptr/core/grammar/conjunctions.py

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConjunctionType(Enum):
    COORDINATING = "coordinating"    # and, but, or, nor, for, so, yet
    SUBORDINATING = "subordinating"  # because, although, since, unless, etc.
    CORRELATIVE = "correlative"      # both...and, either...or, neither...nor, not only...but also


@dataclass
class ConjunctionInfo:
    type: ConjunctionType
    conjunction: str
    left_clause: str
    right_clause: str
    is_parallel: bool = False
    is_contrast: bool = False
    is_cause: bool = False


class ConjunctionHandler:
    """
    Handles all conjunctions (connecting words)
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Coordinating conjunctions (FANBOYS)
        self.coordinating = {
            "for": {"type": "reason", "meaning": "အကြောင်းမှာ"},
            "and": {"type": "addition", "meaning": "နှင့်"},
            "nor": {"type": "negative_addition", "meaning": "လည်းမဟုတ်"},
            "but": {"type": "contrast", "meaning": "သို့သော်"},
            "or": {"type": "alternative", "meaning": "သို့မဟုတ်"},
            "yet": {"type": "contrast", "meaning": "သို့ပေမယ့်"},
            "so": {"type": "result", "meaning": "ဒါကြောင့်"}
        }
        
        # Subordinating conjunctions
        self.subordinating = {
            "because": {"type": "cause", "meaning": "ဘာလို့လဲဆိုတော့"},
            "since": {"type": "cause_time", "meaning": "ကတည်းက / သောကြောင့်"},
            "although": {"type": "concession", "meaning": "သော်လည်း"},
            "though": {"type": "concession", "meaning": "သော်လည်း"},
            "whereas": {"type": "contrast", "meaning": "လျှင်"},
            "while": {"type": "time_contrast", "meaning": "နေစဉ် / လျှင်"},
            "unless": {"type": "condition", "meaning": "မဟုတ်လျှင်"},
            "if": {"type": "condition", "meaning": "အကယ်၍"},
            "when": {"type": "time", "meaning": "သည့်အခါ"},
            "where": {"type": "place", "meaning": "သည့်နေရာ"},
            "that": {"type": "nominal", "meaning": "ဟူသော"}
        }
        
        # Correlative conjunctions
        self.correlative = [
            ("both", "and"),
            ("either", "or"),
            ("neither", "nor"),
            ("not only", "but also"),
            ("whether", "or")
        ]
    
    def detect_conjunctions(self, tokens: List[str]) -> List[ConjunctionInfo]:
        """
        Detect all conjunctions in sentence
        """
        results = []
        sentence = " ".join(tokens)
        
        # Check coordinating conjunctions
        for conj, info in self.coordinating.items():
            if f" {conj} " in f" {sentence} ":  # Word boundary
                parts = sentence.split(f" {conj} ", 1)
                if len(parts) == 2:
                    results.append(ConjunctionInfo(
                        type=ConjunctionType.COORDINATING,
                        conjunction=conj,
                        left_clause=parts[0].strip(),
                        right_clause=parts[1].strip(),
                        is_contrast=(info["type"] == "contrast"),
                        is_cause=(info["type"] == "reason")
                    ))
        
        # Check subordinating conjunctions
        for conj, info in self.subordinating.items():
            if f"{conj} " in sentence:
                parts = sentence.split(f"{conj} ", 1)
                if len(parts) == 2:
                    results.append(ConjunctionInfo(
                        type=ConjunctionType.SUBORDINATING,
                        conjunction=conj,
                        left_clause=parts[0].strip(),
                        right_clause=parts[1].strip(),
                        is_cause=(info["type"] == "cause")
                    ))
        
        return results
    
    def split_clauses(self, sentence: str, conjunction: str) -> Tuple[str, str]:
        """
        Split sentence into clauses based on conjunction
        """
        if conjunction not in sentence:
            return sentence, ""
        
        parts = sentence.split(conjunction, 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return sentence, ""
    
    def get_meaning(self, conjunction: str) -> str:
        """Get Myanmar meaning of conjunction"""
        if conjunction in self.coordinating:
            return self.coordinating[conjunction]["meaning"]
        if conjunction in self.subordinating:
            return self.subordinating[conjunction]["meaning"]
        return ""
