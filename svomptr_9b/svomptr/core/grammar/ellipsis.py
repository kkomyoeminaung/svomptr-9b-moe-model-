# /svomptr_9b/svomptr/core/grammar/ellipsis.py

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class EllipsisInfo:
    has_ellipsis: bool
    omitted_words: List[str]
    context: Optional[str]  # What was omitted
    position: str  # "beginning", "middle", "end", "answer"


class EllipsisHandler:
    """
    Handles ellipsis (omitted words)
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Short answers that imply ellipsis
        self.short_answers = [
            "yes", "no", "yeah", "nope", "yep",
            "ဟုတ်", "မဟုတ်", "အေး", "ဟုတ်ကဲ့",
            "ok", "okay", "sure", "absolutely", "definitely"
        ]
        
        # Patterns where words are typically omitted
        self.omission_patterns = {
            "subject": ["I", "you", "he", "she", "it", "we", "they"],
            "auxiliary": ["is", "am", "are", "was", "were", "do", "does", "did", "have", "has"],
            "verb": ["go", "come", "do", "make", "take", "get"]
        }
    
    def detect_ellipsis(self, sentence: str, context: str = "") -> EllipsisInfo:
        """
        Detect ellipsis in sentence
        """
        sentence_lower = sentence.lower()
        words = sentence.split()
        
        # Case 1: Short answer ellipsis
        if len(words) == 1 and words[0].lower() in self.short_answers:
            return EllipsisInfo(
                has_ellipsis=True,
                omitted_words=["I", "think", "that", "yes"],
                context=context,
                position="answer"
            )
        
        # Case 2: Missing subject (imperative implied)
        if len(words) >= 1 and words[0] not in self.omission_patterns["subject"]:
            # Check if first word is a verb (likely imperative)
            if self._is_verb(words[0]):
                return EllipsisInfo(
                    has_ellipsis=True,
                    omitted_words=["you"],
                    context="imperative",
                    position="beginning"
                )
        
        # Case 3: Missing verb in comparison
        if "than" in sentence_lower and len(words) >= 3:
            # "He is taller than me" (omitted "am")
            return EllipsisInfo(
                has_ellipsis=True,
                omitted_words=["am", "is", "are"],
                context="comparison",
                position="end"
            )
        
        return EllipsisInfo(False, [], None, "")
    
    def _is_verb(self, word: str) -> bool:
        """Simple verb detection"""
        verbs = ["go", "come", "sit", "stand", "run", "eat", "drink", 
                "read", "write", "help", "stop", "start", "wait", "သွား", "လာ", "ထိုင်"]
        return word.lower() in verbs
    
    def restore_ellipsis(self, sentence: str, context: str = "") -> str:
        """
        Attempt to restore omitted words
        """
        info = self.detect_ellipsis(sentence, context)
        
        if not info.has_ellipsis:
            return sentence
        
        if info.position == "answer":
            if sentence.lower() == "yes":
                return "Yes, that is correct."
            elif sentence.lower() == "no":
                return "No, that is not correct."
        
        if info.position == "beginning" and "you" in info.omitted_words:
            return f"You {sentence}"
        
        return sentence
