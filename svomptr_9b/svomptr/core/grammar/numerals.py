# /svomptr_9b/svomptr/core/grammar/numerals.py

"""
Numerals Handler - Cardinal and Ordinal numbers
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class NumeralType(Enum):
    CARDINAL = "cardinal"    # one, two, three
    ORDINAL = "ordinal"      # first, second, third
    MULTIPLICATIVE = "multiplicative"  # once, twice, thrice
    NUMERIC = "numeric"      # 1, 2, 3


@dataclass
class NumeralInfo:
    type: NumeralType
    word: str
    value: int
    position: int
    modifies: Optional[str] = None
    is_approximate: bool = False
    myanmar_equivalent: str = ""


class NumeralsHandler:
    """
    Handles all numeral expressions
    """
    
    def __init__(self):
        # ============================================================
        # Cardinal numbers (1-100)
        # ============================================================
        self.cardinal_words = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
            14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
            18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty",
            40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
            80: "eighty", 90: "ninety", 100: "one hundred",
            1000: "one thousand", 1000000: "one million"
        }
        
        # Reverse lookup (word → value)
        self.cardinal_reverse = {v: k for k, v in self.cardinal_words.items()}
        
        # Special words
        self.cardinal_reverse["a hundred"] = 100
        self.cardinal_reverse["a thousand"] = 1000
        self.cardinal_reverse["a million"] = 1000000
        
        # ============================================================
        # Ordinal numbers
        # ============================================================
        self.ordinal_words = {
            1: "first", 2: "second", 3: "third", 4: "fourth",
            5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
            9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth",
            20: "twentieth", 30: "thirtieth", 40: "fortieth",
            50: "fiftieth", 60: "sixtieth", 70: "seventieth",
            80: "eightieth", 90: "ninetieth", 100: "hundredth",
            1000: "thousandth", 1000000: "millionth"
        }
        
        self.ordinal_reverse = {v: k for k, v in self.ordinal_words.items()}
        
        # ============================================================
        # Multiplicative numbers
        # ============================================================
        self.multiplicative = {
            "once": 1, "twice": 2, "thrice": 3, "four times": 4, "five times": 5
        }
        
        # ============================================================
        # Myanmar numerals
        # ============================================================
        self.myanmar_cardinal = {
            "သုည": 0, "တစ်": 1, "နှစ်": 2, "သုံး": 3, "လေး": 4,
            "ငါး": 5, "ခြောက်": 6, "ခုနစ်": 7, "ရှစ်": 8, "ကိုး": 9,
            "ဆယ်": 10, "ဆယ့်တစ်": 11, "ဆယ့်နှစ်": 12
        }
        
        self.myanmar_ordinal = {
            "ပထမ": 1, "ဒုတိယ": 2, "တတိယ": 3, "စတုတ္ထ": 4,
            "ပဉ္စမ": 5, "ဆဋ္ဌမ": 6, "သတ္တမ": 7, "အဋ္ဌမ": 8,
            "နဝမ": 9, "ဒသမ": 10
        }
        
        # ============================================================
        # Myanmar number words
        # ============================================================
        self.myanmar_number_words = {
            "တစ်ဆယ်": 10, "ရာ": 100, "ထောင်": 1000, "သောင်း": 10000,
            "သိန်း": 100000, "သန်း": 1000000
        }
    
    def detect_numerals(self, tokens: List[str]) -> List[NumeralInfo]:
        """Detect all numerals in sentence"""
        numerals = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            # ========================================================
            # 1. Numeric digits (1, 2, 3)
            # ========================================================
            if token.isdigit():
                numerals.append(NumeralInfo(
                    type=NumeralType.NUMERIC,
                    word=token,
                    value=int(token),
                    position=i,
                    modifies=tokens[i+1] if i+1 < len(tokens) else None
                ))
            
            # ========================================================
            # 2. Cardinal words (one, two, three)
            # ========================================================
            elif token_lower in self.cardinal_reverse:
                numerals.append(NumeralInfo(
                    type=NumeralType.CARDINAL,
                    word=token,
                    value=self.cardinal_reverse[token_lower],
                    position=i,
                    modifies=tokens[i+1] if i+1 < len(tokens) else None,
                    myanmar_equivalent=self._get_myanmar_cardinal(self.cardinal_reverse[token_lower])
                ))
            
            # ========================================================
            # 3. Ordinal words (first, second, third)
            # ========================================================
            elif token_lower in self.ordinal_reverse:
                numerals.append(NumeralInfo(
                    type=NumeralType.ORDINAL,
                    word=token,
                    value=self.ordinal_reverse[token_lower],
                    position=i,
                    modifies=tokens[i+1] if i+1 < len(tokens) else None,
                    myanmar_equivalent=self._get_myanmar_ordinal(self.ordinal_reverse[token_lower])
                ))
            
            # ========================================================
            # 4. Multiplicative (once, twice, thrice)
            # ========================================================
            elif token_lower in self.multiplicative:
                numerals.append(NumeralInfo(
                    type=NumeralType.MULTIPLICATIVE,
                    word=token,
                    value=self.multiplicative[token_lower],
                    position=i,
                    modifies=None
                ))
            
            # ========================================================
            # 5. Myanmar numerals
            # ========================================================
            elif token_lower in self.myanmar_cardinal:
                numerals.append(NumeralInfo(
                    type=NumeralType.CARDINAL,
                    word=token,
                    value=self.myanmar_cardinal[token_lower],
                    position=i,
                    modifies=tokens[i+1] if i+1 < len(tokens) else None,
                    myanmar_equivalent=token_lower
                ))
            
            elif token_lower in self.myanmar_ordinal:
                numerals.append(NumeralInfo(
                    type=NumeralType.ORDINAL,
                    word=token,
                    value=self.myanmar_ordinal[token_lower],
                    position=i,
                    modifies=tokens[i+1] if i+1 < len(tokens) else None,
                    myanmar_equivalent=token_lower
                ))
        
        return numerals
    
    def _get_myanmar_cardinal(self, value: int) -> str:
        """Get Myanmar cardinal number"""
        myanmar_numbers = {
            1: "တစ်", 2: "နှစ်", 3: "သုံး", 4: "လေး", 5: "ငါး",
            6: "ခြောက်", 7: "ခုနစ်", 8: "ရှစ်", 9: "ကိုး", 10: "ဆယ်"
        }
        return myanmar_numbers.get(value, str(value))
    
    def _get_myanmar_ordinal(self, value: int) -> str:
        """Get Myanmar ordinal number"""
        myanmar_ordinals = {
            1: "ပထမ", 2: "ဒုတိယ", 3: "တတိယ", 4: "စတုတ္ထ",
            5: "ပဉ္စမ", 6: "ဆဋ္ဌမ", 7: "သတ္တမ", 8: "အဋ္ဌမ",
            9: "နဝမ", 10: "ဒသမ"
        }
        return myanmar_ordinals.get(value, str(value))
    
    def get_numeral_rules(self) -> Dict:
        """Get numeral usage rules"""
        return {
            "cardinal": {
                "usage": "အရေအတွက်ပြရန် (ဘယ်နှစ်ခု)",
                "examples": ["one apple", "three cars", "ten people"]
            },
            "ordinal": {
                "usage": "အစီအစဉ်ပြရန် (ဘယ်နှစ်ယောက်မြောက်)",
                "examples": ["first place", "second chance", "third time"]
            },
            "numeric": {
                "usage": "နံပါတ်များ",
                "examples": ["page 5", "room 101", "year 2024"]
            }
        }
    
    def convert_to_words(self, number: int, ordinal: bool = False) -> str:
        """Convert number to English words"""
        if ordinal:
            return self.ordinal_words.get(number, str(number))
        return self.cardinal_words.get(number, str(number))
