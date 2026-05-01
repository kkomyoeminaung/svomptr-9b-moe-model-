# /svomptr_9b/svomptr/core/grammar/negation.py

from typing import List, Tuple, Optional


class NegationHandler:
    """
    Handles all negation patterns
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Standard negation words
        self.negation_words = {
            "not": "standard",
            "never": "time_negation",
            "no": "determiner_negation",
            "none": "pronoun_negation",
            "nobody": "pronoun_negation",
            "nothing": "pronoun_negation",
            "nowhere": "place_negation",
            "neither": "paired_negation",
            "nor": "paired_negation"
        }
        
        # Contracted forms
        self.contractions = {
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "won't": "will not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "mightn't": "might not",
            "mustn't": "must not"
        }
        
        # Myanmar negation
        self.myanmar_negation = ["မ", "မဟုတ်", "မရှိ", "မဖြစ်", "မနိုင်"]
    
    def detect_negation(self, tokens: List[str]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect negation in sentence
        Returns: (has_negation, negated_word, negation_type)
        """
        sentence = " ".join(tokens).lower()
        words = sentence.split()
        
        # Check contractions first
        for contraction, expanded in self.contractions.items():
            if contraction in sentence:
                # Find what is being negated
                for i, word in enumerate(words):
                    if word == contraction:
                        negated = words[i-1] if i > 0 else None
                        return True, negated, "contraction"
        
        # Check standard negation words
        for neg_word in self.negation_words:
            if neg_word in words:
                # Find what is being negated
                for i, w in enumerate(words):
                    if w == neg_word:
                        negated = words[i+1] if i + 1 < len(words) else None
                        return True, negated, self.negation_words[neg_word]
        
        # Check Myanmar negation
        for neg in self.myanmar_negation:
            if neg in sentence:
                return True, None, "myanmar"
        
        return False, None, None
    
    def get_affirmative_form(self, negated_sentence: str) -> str:
        """
        Convert negative to affirmative (simplified)
        Example: "I do not like it" → "I like it"
        """
        # Remove negation words
        result = negated_sentence
        for neg in self.negation_words:
            result = result.replace(f" {neg} ", " ")
        for contraction, expanded in self.contractions.items():
            result = result.replace(contraction, "")
        return result.strip()
    
    def get_negative_form(self, affirmative_sentence: str) -> str:
        """
        Convert affirmative to negative (simplified)
        Example: "I like it" → "I do not like it"
        """
        words = affirmative_sentence.split()
        if len(words) < 2:
            return f"{affirmative_sentence} not"
        
        # Simple rule: add "not" after first auxiliary verb or do/does/did
        aux_verbs = ["am", "is", "are", "was", "were", 
                     "have", "has", "had", "will", "would", "should", "could", "may", "might"]
        
        for i, word in enumerate(words):
            if word in aux_verbs:
                words.insert(i + 1, "not")
                return " ".join(words)
        
        # Add "do not" before verb
        return f"Do not {affirmative_sentence}"
