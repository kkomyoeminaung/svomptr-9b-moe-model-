# /svomptr_9b/svomptr/core/grammar/appositives.py

"""
Appositives and Parentheticals Handler
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AppositiveInfo:
    is_appositive: bool
    main_noun: str
    appositive: str
    position: int
    is_restrictive: bool  # Essential or non-essential


@dataclass
class ParentheticalInfo:
    is_parenthetical: bool
    text: str
    position: int
    type: str  # "introductory", "interrupting", "concluding"


class AppositiveHandler:
    """Handles appositives (noun phrases that rename another noun)"""
    
    def __init__(self):
        # Punctuation marks that indicate appositives
        self.appositive_markers = [",", ":", "-", "("]
    
    def detect_appositives(self, tokens: List[str]) -> List[AppositiveInfo]:
        """
        Detect appositives in sentence
        Example: "My brother, a doctor, lives in Yangon"
        """
        appositives = []
        sentence = " ".join(tokens)
        
        # Pattern 1: Noun, appositive, (with commas)
        if ", " in sentence:
            parts = sentence.split(", ")
            if len(parts) >= 3:
                # Format: noun, appositive, rest of sentence
                main_noun_parts = parts[0].split()
                if main_noun_parts:
                    main_noun = main_noun_parts[-1]  # Last word before comma
                    appositive = parts[1]
                    appositives.append(AppositiveInfo(
                        is_appositive=True,
                        main_noun=main_noun,
                        appositive=appositive,
                        position=sentence.find(","),
                        is_restrictive=False  # Non-restrictive (with commas)
                    ))
        
        # Pattern 2: Noun (appositive)
        if "(" in sentence and ")" in sentence:
            start = sentence.find("(")
            end = sentence.find(")")
            if start > 0 and end > start:
                main_noun_parts = sentence[:start].split()
                if main_noun_parts:
                    main_noun = main_noun_parts[-1].strip()
                    appositive = sentence[start+1:end]
                    appositives.append(AppositiveInfo(
                        is_appositive=True,
                        main_noun=main_noun,
                        appositive=appositive,
                        position=start,
                        is_restrictive=True  # Restrictive (parentheses)
                    ))
        
        return appositives
    
    def detect_parentheticals(self, tokens: List[str]) -> List[ParentheticalInfo]:
        """
        Detect parentheticals (inserted remarks)
        Example: "This, I believe, is correct"
        """
        parentheticals = []
        sentence = " ".join(tokens)
        
        # Pattern 1: Introductory parenthetical (word, sentence)
        introductory = ["well", "actually", "honestly", "frankly", "basically"]
        words = sentence.split()
        if words and words[0].lower().strip(",") in introductory:
            parentheticals.append(ParentheticalInfo(
                is_parenthetical=True,
                text=words[0].strip(","),
                position=0,
                type="introductory"
            ))
        
        # Pattern 2: Interrupting parenthetical (sentence , phrase , sentence)
        if ", " in sentence:
            parts = sentence.split(", ")
            if len(parts) >= 3:
                # Check if middle part is a parenthetical
                middle = parts[1]
                parentheticals.append(ParentheticalInfo(
                    is_parenthetical=True,
                    text=middle,
                    position=sentence.find(", ") + 2,
                    type="interrupting"
                ))
        
        return parentheticals
    
    def remove_parentheticals(self, sentence: str) -> str:
        """Remove parentheticals for core parsing"""
        import re
        # Remove parentheticals between commas (non-restrictive)
        sentence = re.sub(r',\s*[^,]+,\s*', ', ', sentence)
        # Remove parentheses content
        sentence = re.sub(r'\([^)]*\)', '', sentence)
        # Remove brackets content
        sentence = re.sub(r'\[[^\]]*\]', '', sentence)
        return sentence.strip()
    
    def get_appositive_rules(self) -> Dict:
        """Get appositive usage rules"""
        return {
            "restrictive": {
                "meaning": "မရှိမဖြစ်လိုအပ်သော (Essential)",
                "punctuation": "စကားခြားမပါ",
                "example": "My friend John is here"
            },
            "non_restrictive": {
                "meaning": "ထပ်ဆောင်းအချက်အလက် (Non-essential)",
                "punctuation": "စကားခြား (,) သုံးသည်",
                "example": "My friend, John, is here"
            }
        }
