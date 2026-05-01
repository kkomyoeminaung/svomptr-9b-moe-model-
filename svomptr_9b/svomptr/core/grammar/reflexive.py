# /svomptr_9b/svomptr/core/grammar/reflexive.py

"""
Reflexive Pronouns Handler
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ReflexiveType(Enum):
    REFLEXIVE = "reflexive"      # He hurt himself
    EMPHATIC = "emphatic"        # I myself did it
    IDIOMATIC = "idiomatic"      # by myself, enjoy yourself


@dataclass
class ReflexiveInfo:
    is_reflexive: bool
    pronoun: str
    type: ReflexiveType
    refers_to: str  # The subject it refers to
    position: int = -1
    meaning: str = ""


class ReflexiveHandler:
    """
    Handles all reflexive pronouns
    """
    
    def __init__(self):
        # ============================================================
        # Reflexive pronouns
        # ============================================================
        self.reflexive_pronouns = {
            "myself": "ကိုယ်တိုင် (1st person singular)",
            "yourself": "ကိုယ်တိုင် (2nd person singular)",
            "yourselves": "ကိုယ်တိုင် (2nd person plural)",
            "himself": "သူ့ကိုယ်သူ (3rd person masculine)",
            "herself": "သူမကိုယ်သူမ (3rd person feminine)",
            "itself": "၎င်းကိုယ်၎င်း (3rd person neuter)",
            "ourselves": "ကိုယ်တိုင် (1st person plural)",
            "themselves": "သူတို့ကိုယ်သူတို့ (3rd person plural)"
        }
        
        # Subject to reflexive mapping
        self.subject_reflexive = {
            "i": "myself",
            "you": "yourself",
            "he": "himself",
            "she": "herself",
            "it": "itself",
            "we": "ourselves",
            "they": "themselves"
        }
        
        # ============================================================
        # Idiomatic reflexive expressions
        # ============================================================
        self.idiomatic_expressions = {
            "by myself": "တစ်ယောက်တည်း",
            "by yourself": "တစ်ယောက်တည်း",
            "by himself": "တစ်ယောက်တည်း",
            "by herself": "တစ်ယောက်တည်း",
            "by itself": "တစ်ခုတည်း",
            "by ourselves": "ကိုယ်တိုင်ကိုယ်ကျ",
            "by themselves": "သူတို့ဘာသာ",
            "enjoy yourself": "ပျော်ပါ",
            "help yourself": "စားပါ (ဧည့်ခံ)",
            "make yourself at home": "အိမ်သဖွယ်နေပါ"
        }
        
        # ============================================================
        # Emphatic reflexive patterns
        # ============================================================
        self.emphatic_patterns = [
            ("i", "myself"), ("you", "yourself"), ("he", "himself"),
            ("she", "herself"), ("it", "itself"), ("we", "ourselves"),
            ("they", "themselves")
        ]
    
    def detect_reflexive(self, tokens: List[str]) -> List[ReflexiveInfo]:
        """Detect all reflexive pronouns in sentence"""
        reflexives = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            if token_lower in self.reflexive_pronouns:
                # ====================================================
                # Determine type: reflexive, emphatic, or idiomatic
                # ====================================================
                
                # Check for idiomatic expression (by myself, enjoy yourself)
                is_idiomatic = False
                meaning = ""
                for expr, m in self.idiomatic_expressions.items():
                    if expr in " ".join(tokens):
                        is_idiomatic = True
                        meaning = m
                        break
                
                if is_idiomatic:
                    reflex_type = ReflexiveType.IDIOMATIC
                else:
                    # Check if it's emphatic (directly after subject)
                    is_emphatic = False
                    for sub, ref in self.subject_reflexive.items():
                        if i > 0 and tokens[i-1].lower() == sub and token_lower == ref:
                            is_emphatic = True
                            break
                    
                    reflex_type = ReflexiveType.EMPHATIC if is_emphatic else ReflexiveType.REFLEXIVE
                
                # Find what it refers to (subject)
                refers_to = self._find_referent(tokens, i)
                
                reflexives.append(ReflexiveInfo(
                    is_reflexive=True,
                    pronoun=token,
                    type=reflex_type,
                    refers_to=refers_to,
                    position=i,
                    meaning=meaning or self.reflexive_pronouns.get(token_lower, "")
                ))
        
        return reflexives
    
    def _find_referent(self, tokens: List[str], pos: int) -> str:
        """Find what the reflexive pronoun refers to"""
        # Look for subject before the reflexive
        for i in range(pos - 1, -1, -1):
            token_lower = tokens[i].lower()
            if token_lower in self.subject_reflexive:
                return token_lower
        
        # Check for proper noun before reflexive
        for i in range(pos - 1, -1, -1):
            if tokens[i][0].isupper():
                return tokens[i]
        
        return "unknown"
    
    def get_reflexive_rules(self) -> Dict:
        """Get reflexive pronoun usage rules"""
        return {
            "reflexive": {
                "description": "ကြိယာ၏အကျိုးကို ခံရသူ (အကြောင်းခံနှင့်တူ)",
                "example": "He hurt himself (သူ့ကိုယ်သူ နာသည်)",
                "meaning": "Subject က object ကို လုပ်ဆောင်သည် (တူညီသူ)"
            },
            "emphatic": {
                "description": "အလေးပေးဖော်ပြရန်",
                "example": "I myself did it (ကျွန်တော် ကိုယ်တိုင်မြင်သည်)",
                "meaning": "ကိုယ်တိုင်"
            },
            "idiomatic": {
                "description": "စကားထူးများ",
                "examples": [
                    "by myself (တစ်ယောက်တည်း)",
                    "enjoy yourself (ပျော်ပါ)",
                    "make yourself at home (အိမ်သဖွယ်နေပါ)"
                ]
            }
        }
    
    def get_possessive_reflexive(self, subject: str) -> str:
        """Convert possessive adjective to reflexive"""
        poss_reflexive = {
            "my": "myself",
            "your": "yourself",
            "his": "himself",
            "her": "herself",
            "its": "itself",
            "our": "ourselves",
            "their": "themselves"
        }
        return poss_reflexive.get(subject.lower(), "")
