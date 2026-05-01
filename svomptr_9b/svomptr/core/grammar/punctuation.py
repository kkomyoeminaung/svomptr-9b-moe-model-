# /svomptr_9b/svomptr/core/grammar/punctuation.py

"""
Punctuation Handler
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PunctuationType(Enum):
    PERIOD = "period"          # .
    COMMA = "comma"            # ,
    QUESTION = "question"      # ?
    EXCLAMATION = "exclamation" # !
    SEMICOLON = "semicolon"    # ;
    COLON = "colon"            # :
    QUOTE_SINGLE = "quote_single"  # '
    QUOTE_DOUBLE = "quote_double"  # "
    PARENTHESIS = "parenthesis"    # ()
    BRACKET = "bracket"            # []
    ELLIPSIS = "ellipsis"          # ...


@dataclass
class PunctuationInfo:
    type: PunctuationType
    char: str
    position: int
    function: str


class PunctuationHandler:
    """Handles all punctuation marks and their functions"""
    
    def __init__(self):
        # Punctuation marks
        self.punctuations = {
            ".": PunctuationType.PERIOD,
            ",": PunctuationType.COMMA,
            "?": PunctuationType.QUESTION,
            "!": PunctuationType.EXCLAMATION,
            ";": PunctuationType.SEMICOLON,
            ":": PunctuationType.COLON,
            "'": PunctuationType.QUOTE_SINGLE,
            '"': PunctuationType.QUOTE_DOUBLE,
            "(": PunctuationType.PARENTHESIS,
            ")": PunctuationType.PARENTHESIS,
            "[": PunctuationType.BRACKET,
            "]": PunctuationType.BRACKET,
            "...": PunctuationType.ELLIPSIS
        }
        
        # Period functions (sentence end vs abbreviation)
        self.abbreviations = [
            "mr.", "mrs.", "ms.", "dr.", "prof.", "gen.", "col.", 
            "e.g.", "i.e.", "etc.", "vs.", "inc.", "co.", "corp."
        ]
    
    def detect_punctuation(self, text: str) -> List[PunctuationInfo]:
        """Detect all punctuation marks in text"""
        punctuations = []
        
        for i, char in enumerate(text):
            if char in self.punctuations:
                punct_type = self.punctuations[char]
                
                # Determine function
                function = self._get_function(text, i, char, punct_type)
                
                punctuations.append(PunctuationInfo(
                    type=punct_type,
                    char=char,
                    position=i,
                    function=function
                ))
        
        return punctuations
    
    def _get_function(self, text: str, pos: int, char: str, punct_type: PunctuationType) -> str:
        """Determine function of punctuation mark"""
        if punct_type == PunctuationType.PERIOD:
            # Check if it's an abbreviation
            for abbr in self.abbreviations:
                if text[pos - len(abbr) + 1:pos + 1].lower() == abbr:
                    return "abbreviation"
            # Check if it's at end of sentence
            if pos == len(text) - 1 or text[pos + 1].isspace():
                return "sentence_end"
            return "decimal_point"
        
        elif punct_type == PunctuationType.COMMA:
            # Determine comma function
            before = text[:pos].split()[-1] if text[:pos].split() else ""
            after = text[pos+1:].split()[0] if text[pos+1:].split() else ""
            
            if before.lower() in ["however", "therefore", "thus", "consequently"]:
                return "conjunctive_adverb"
            elif after.lower() in ["and", "but", "or", "nor", "for", "so", "yet"]:
                return "coordinating_conjunction"
            else:
                return "clause_separator"
        
        elif punct_type == PunctuationType.QUESTION:
            return "question_marker"
        
        elif punct_type == PunctuationType.EXCLAMATION:
            return "exclamation_marker"
        
        elif punct_type == PunctuationType.SEMICOLON:
            return "related_clauses"
        
        elif punct_type == PunctuationType.COLON:
            return "list_or_explanation"
        
        return "unknown"
    
    def get_punctuation_rules(self) -> Dict:
        """Get punctuation usage rules"""
        return {
            "period": {
                "uses": ["စာကြောင်းဆုံး", "အတိုကောက်", "ဒဿမ"],
                "examples": ["I am happy.", "Dr. Smith", "3.14"]
            },
            "comma": {
                "uses": ["စာကြောင်းခွဲရန်", "list အတွက်", "conjunction ရှေ့"],
                "examples": ["I like cats, dogs, and birds", "When I came, he left"]
            }
        }
