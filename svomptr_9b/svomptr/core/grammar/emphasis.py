# /svomptr_9b/svomptr/core/grammar/emphasis.py

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class EmphasisType(Enum):
    CLEFT = "cleft"              # "It was John who called"
    PSEUDO_CLEFT = "pseudo_cleft"  # "What I need is rest"
    INVERSION = "inversion"      # "Never have I seen such beauty"
    EMPHATIC_DO = "emphatic_do"  # "I do love you"
    FRONTING = "fronting"        # "This I believe"


@dataclass
class EmphasisInfo:
    has_emphasis: bool
    type: Optional[EmphasisType]
    focused_element: Optional[str]
    original_form: Optional[str]


class EmphasisHandler:
    """
    Handles emphasis constructions
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Cleft sentence patterns (It + be + focused element + that/who clause)
        self.cleft_pattern = {
            "markers": ["it", "it's", "it was"],
            "connectors": ["that", "who", "whom", "which", "where", "when"]
        }
        
        # Pseudo-cleft patterns (What/All/The thing + be clause)
        self.pseudo_cleft_pattern = {
            "starters": ["what", "all", "the thing", "the one"],
            "be_verbs": ["is", "was", "are", "were"]
        }
        
        # Inversion patterns (Negative/restrictive + auxiliary + subject)
        self.inversion_triggers = [
            "never", "rarely", "seldom", "hardly", "scarcely",
            "not only", "no sooner", "only then", "only after",
            "not until", "in no way", "under no circumstances"
        ]
        
        # Emphatic do/does/did (used for emphasis in positive statements)
        self.emphatic_auxiliaries = ["do", "does", "did"]
    
    def detect_emphasis(self, tokens: List[str]) -> EmphasisInfo:
        """
        Detect emphasis constructions
        """
        sentence = " ".join(tokens).lower()
        
        # Check for cleft sentence
        if self._is_cleft(sentence):
            focused = self._extract_cleft_focus(tokens)
            return EmphasisInfo(
                has_emphasis=True,
                type=EmphasisType.CLEFT,
                focused_element=focused,
                original_form=self._transform_cleft_to_normal(sentence)
            )
        
        # Check for inversion
        if self._is_inversion(tokens):
            return EmphasisInfo(
                has_emphasis=True,
                type=EmphasisType.INVERSION,
                focused_element=tokens[0] if tokens else None,
                original_form=self._transform_inversion_to_normal(tokens)
            )
        
        # Check for emphatic do
        for i, token in enumerate(tokens):
            if token.lower() in self.emphatic_auxiliaries and i + 1 < len(tokens):
                # Check if it's emphatic (not a question)
                if not sentence.endswith("?"):
                    return EmphasisInfo(
                        has_emphasis=True,
                        type=EmphasisType.EMPHATIC_DO,
                        focused_element=tokens[i + 1] if i + 1 < len(tokens) else None,
                        original_form=self._remove_emphatic_do(tokens)
                    )
        
        return EmphasisInfo(False, None, None, None)
    
    def _is_cleft(self, sentence: str) -> bool:
        """Check if sentence is a cleft sentence"""
        sentence_lower = sentence.lower()
        for marker in self.cleft_pattern["markers"]:
            if sentence_lower.startswith(marker):
                for connector in self.cleft_pattern["connectors"]:
                    if f" {connector} " in sentence_lower:
                        return True
        return False
    
    def _is_inversion(self, tokens: List[str]) -> bool:
        """Check if sentence has inversion"""
        if not tokens:
            return False
        first = tokens[0].lower()
        return first in self.inversion_triggers
    
    def _extract_cleft_focus(self, tokens: List[str]) -> Optional[str]:
        """Extract focused element from cleft sentence"""
        # Simplified extraction
        for i, token in enumerate(tokens):
            if token.lower() in self.cleft_pattern["connectors"]:
                if i > 0:
                    return tokens[i - 1] if i - 1 >= 0 else None
        return None
    
    def _transform_cleft_to_normal(self, cleft_sentence: str) -> str:
        """Transform cleft sentence to normal form (simplified)"""
        # "It was John who called" → "John called"
        # Simplified implementation
        parts = cleft_sentence.lower().split()
        for i, part in enumerate(parts):
            if part in self.cleft_pattern["connectors"]:
                subject = parts[i - 1] if i - 1 >= 0 else ""
                rest = " ".join(parts[i + 1:])
                return f"{subject} {rest}"
        return cleft_sentence
    
    def _transform_inversion_to_normal(self, inverted_tokens: List[str]) -> str:
        """Transform inverted sentence to normal form"""
        # "Never have I seen" → "I have never seen"
        if len(inverted_tokens) >= 4:
            return f"{inverted_tokens[2]} {inverted_tokens[1]} {inverted_tokens[0]} {inverted_tokens[3]}"
        return " ".join(inverted_tokens)
    
    def _remove_emphatic_do(self, tokens: List[str]) -> str:
        """Remove emphatic do/does/did"""
        result = []
        for token in tokens:
            if token.lower() not in self.emphatic_auxiliaries:
                result.append(token)
        return " ".join(result)
