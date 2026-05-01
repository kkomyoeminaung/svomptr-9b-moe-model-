# /svomptr_9b/svomptr/core/grammar/reported_speech.py

from typing import Tuple, Optional, List, Dict


class ReportedSpeechHandler:
    """
    Handles reported speech (direct and indirect)
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Reporting verbs
        self.reporting_verbs = {
            "say": "said", "tell": "told", "ask": "asked",
            "explain": "explained", "state": "stated", "remark": "remarked",
            "mention": "mentioned", "claim": "claimed", "declare": "declared",
            "ပြော": "ပြောသည်", "မေး": "မေးသည်", "ရှင်း": "ရှင်းသည်"
        }
        
        # Tense changes for indirect speech
        self.tense_changes = {
            "present": "past",
            "present continuous": "past continuous",
            "present perfect": "past perfect",
            "past": "past perfect",
            "will": "would",
            "can": "could",
            "may": "might",
            "must": "had to"
        }
        
        # Pronoun changes
        self.pronoun_changes = {
            "I": "he/she", "we": "they", "you": "he/she/they",
            "me": "him/her", "us": "them", "my": "his/her",
            "our": "their", "mine": "his/hers"
        }
    
    def detect_reported_speech(self, tokens: List[str]) -> Tuple[bool, Optional[str], Optional[str], bool]:
        """
        Detect reported speech
        Returns: (is_reported, reporting_verb, reported_clause, is_indirect)
        """
        if len(tokens) < 3:
            return False, None, None, False
        
        # Check for direct speech (quotation marks)
        for i, token in enumerate(tokens):
            if token.startswith('"') or token.startswith("'") or token.startswith('"'):
                # Find closing quote
                is_direct = True
                # Extract quoted part
                quoted_parts = []
                for j in range(i, len(tokens)):
                    quoted_parts.append(tokens[j])
                    if tokens[j].endswith('"') or tokens[j].endswith("'"):
                        break
                quoted_speech = " ".join(quoted_parts).strip('"').strip("'")
                reporting_verb = tokens[i-1] if i > 0 else None
                return True, reporting_verb, quoted_speech, False
        
        # Check for indirect speech (that-clause)
        for i, token in enumerate(tokens):
            if token.lower() in self.reporting_verbs:
                if i + 1 < len(tokens) and tokens[i + 1].lower() == "that":
                    reported = " ".join(tokens[i + 2:])
                    return True, token, reported, True
        
        return False, None, None, False
    
    def transform_to_indirect(self, direct_speech: str, reporting_verb: str = "said", 
                              reporting_subject: str = "he") -> str:
        """
        Transform direct speech to indirect speech
        Example: "I am tired" → He said that he was tired
        """
        # Simplified transformation
        # Remove quotes
        speech = direct_speech.strip('"').strip("'")
        
        # Change pronoun (simplified)
        speech = speech.replace("I", "he/she", 1)
        
        # Change tense (simplified - would need full parser for accuracy)
        if "am" in speech:
            speech = speech.replace("am", "was")
        elif "is" in speech:
            speech = speech.replace("is", "was")
        elif "are" in speech:
            speech = speech.replace("are", "were")
        elif "will" in speech:
            speech = speech.replace("will", "would")
        
        return f"{reporting_subject} {reporting_verb} that {speech}"
    
    def transform_to_direct(self, indirect_speech: str, reporting_verb: str = "said") -> str:
        """
        Transform indirect speech to direct speech (simplified)
        """
        # Remove "that"
        if "that" in indirect_speech:
            parts = indirect_speech.split("that", 1)
            if len(parts) == 2:
                statement = parts[1].strip()
                return f'"{statement}"'
        return f'"{indirect_speech}"'
