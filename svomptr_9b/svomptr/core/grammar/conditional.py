# /svomptr_9b/svomptr/core/grammar/conditional.py

from enum import Enum
from typing import Dict, List, Optional, Tuple


class ConditionalType(Enum):
    ZERO = "zero"      # If + present, present (General truth)
    FIRST = "first"    # If + present, will (Real possibility)
    SECOND = "second"  # If + past, would (Unreal present)
    THIRD = "third"    # If + past perfect, would have (Unreal past)
    MIXED = "mixed"    # Mixed time references


class ConditionalHandler:
    """
    Handles all conditional sentences (If-clauses)
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Conditional markers
        self.conditional_markers = [
            "if", "unless", "provided that", "providing that",
            "as long as", "on condition that", "အကယ်၍", "သို့ဆိုလျှင်"
        ]
        
        # Patterns for each type
        self.patterns = {
            ConditionalType.ZERO: {
                "pattern": "If + present simple, present simple",
                "meaning": "အမြဲတမ်းမှန်သော အချက် (General truth)",
                "example": "If you heat ice, it melts"
            },
            ConditionalType.FIRST: {
                "pattern": "If + present simple, will + base verb",
                "meaning": "ဖြစ်နိုင်သော အနာဂတ်အခြေ (Real possibility)",
                "example": "If it rains, I will stay home"
            },
            ConditionalType.SECOND: {
                "pattern": "If + past simple, would + base verb",
                "meaning": "လက်ရှိမဖြစ်နိုင်သော အခြေ (Unreal present)",
                "example": "If I were rich, I would travel"
            },
            ConditionalType.THIRD: {
                "pattern": "If + past perfect, would have + past participle",
                "meaning": "အတိတ်က မဖြစ်ခဲ့သော အခြေ (Unreal past)",
                "example": "If I had known, I would have come"
            }
        }
    
    def detect_conditional(self, sentence: str) -> Tuple[bool, Optional[ConditionalType], Optional[str], Optional[str]]:
        """
        Detect conditional sentence
        Returns: (is_conditional, type, condition_clause, result_clause)
        """
        sentence_lower = sentence.lower()
        
        for marker in self.conditional_markers:
            if marker in sentence_lower:
                parts = sentence_lower.split(marker, 1)
                if len(parts) == 2:
                    # Extract condition and result
                    remaining = parts[1].strip()
                    
                    # Split at comma or "then"
                    if ", " in remaining or " then " in remaining:
                        if ", " in remaining:
                            condition, result = remaining.split(", ", 1)
                        else:
                            condition, result = remaining.split(" then ", 1)
                        
                        # Determine type
                        cond_type = self._determine_type(condition, result)
                        return True, cond_type, condition.strip(), result.strip()
                    else:
                        return True, None, remaining, None
        
        return False, None, None, None
    
    def _determine_type(self, condition: str, result: str) -> ConditionalType:
        """Determine conditional type from clause patterns"""
        # Type 0: present + present
        if self._has_present(condition) and self._has_present(result):
            return ConditionalType.ZERO
        
        # Type 1: present + will
        if self._has_present(condition) and "will" in result:
            return ConditionalType.FIRST
        
        # Type 2: past + would
        if self._has_past(condition) and "would" in result and "have" not in result:
            return ConditionalType.SECOND
        
        # Type 3: past perfect + would have
        if ("had" in condition or "had" in condition) and "would have" in result:
            return ConditionalType.THIRD
        
        return ConditionalType.MIXED
    
    def _has_present(self, clause: str) -> bool:
        """Check if clause has present tense markers"""
        present_verbs = ["is", "am", "are", "do", "does", "has", "have", 
                         "go", "goes", "come", "comes", "eat", "eats"]
        return any(v in clause for v in present_verbs)
    
    def _has_past(self, clause: str) -> bool:
        """Check if clause has past tense markers"""
        past_verbs = ["was", "were", "did", "had", "went", "came", "ate"]
        return any(v in clause for v in past_verbs) or clause.endswith("ed") or "ed " in clause
    
    def get_explanation(self, cond_type: ConditionalType) -> str:
        """Get Myanmar explanation for conditional type"""
        explanations = {
            ConditionalType.ZERO: "အမြဲမှန်သော အချက်အလက် (သိပ္ပံဆိုင်ရာ ဖြစ်ရပ်မှန်)",
            ConditionalType.FIRST: "ဖြစ်နိုင်ခြေရှိသော အနာဂတ်အခြေအနေ",
            ConditionalType.SECOND: "လက်ရှိမဖြစ်နိုင်သော အနာဂတ်အခြေအနေ (အိပ်မက်များ)",
            ConditionalType.THIRD: "အတိတ်က မဖြစ်ခဲ့သော အခြေအနေ (နောင်တရဖွယ်)",
            ConditionalType.MIXED: "ရောထွေးနေသော အချိန်ရည်ညွှန်းချက်များ"
        }
        return explanations.get(cond_type, "အခြေအနေပြဝါကျ")
