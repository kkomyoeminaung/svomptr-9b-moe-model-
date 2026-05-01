# /svomptr_9b/svomptr/core/grammar/tag_questions.py

"""
Tag Questions Handler
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TagQuestionInfo:
    is_tag_question: bool
    main_statement: str
    tag: str
    tag_auxiliary: str
    tag_pronoun: str
    is_positive_tag: bool  # True for positive tag (don't you?)


class TagQuestionHandler:
    """Handles tag questions (You like it, don't you?)"""
    
    def __init__(self):
        # Auxiliary verbs and their negative forms
        self.auxiliary_verbs = {
            "is": "isn't", "are": "aren't", "am": "amn't",
            "was": "wasn't", "were": "weren't",
            "do": "don't", "does": "doesn't", "did": "didn't",
            "have": "haven't", "has": "hasn't", "had": "hadn't",
            "will": "won't", "would": "wouldn't",
            "shall": "shan't", "should": "shouldn't",
            "can": "can't", "could": "couldn't",
            "may": "mayn't", "might": "mightn't",
            "must": "mustn't"
        }
        
        # Pronouns for tags
        self.tag_pronouns = {
            "i": "i", "you": "you", "he": "he",
            "she": "she", "it": "it", "we": "we",
            "they": "they",
            # Special cases
            "there": "there", "this": "it", "that": "it"
        }
    
    def detect_tag_question(self, sentence: str) -> TagQuestionInfo:
        """
        Detect if sentence has a tag question
        Example: "You like coffee, don't you?"
        """
        sentence_lower = sentence.lower()
        
        # Look for comma followed by tag
        if ", " in sentence_lower and "?" in sentence_lower:
            parts = sentence_lower.rsplit(", ", 1)
            if len(parts) == 2:
                main_part = parts[0]
                tag_part = parts[1]
                
                # Extract tag components
                tag_words = tag_part.split()
                if len(tag_words) >= 2:
                    tag_aux = tag_words[0]
                    tag_pron = tag_words[1].strip("?")
                    
                    # Determine if positive or negative tag
                    is_positive_tag = "'t" not in tag_aux and "not" not in tag_aux
                    
                    return TagQuestionInfo(
                        is_tag_question=True,
                        main_statement=main_part,
                        tag=tag_part.strip("?"),
                        tag_auxiliary=tag_aux,
                        tag_pronoun=tag_pron,
                        is_positive_tag=is_positive_tag
                    )
        
        return TagQuestionInfo(False, "", "", "", "", False)
    
    def get_tag_question_rules(self) -> Dict:
        """Get tag question formation rules"""
        return {
            "basic_rule": {
                "description": "Positive statement → negative tag / Negative statement → positive tag",
                "example1": "You are happy, aren't you?",
                "example2": "You aren't sad, are you?"
            },
            "with_auxiliary": {
                "rule": "Same auxiliary verb in tag",
                "example": "She can swim, can't she?"
            },
            "without_auxiliary": {
                "rule": "Use do/does/did in tag",
                "example": "You like it, don't you?"
            }
        }
