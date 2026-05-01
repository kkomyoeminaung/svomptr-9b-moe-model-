# /svomptr_9b/svomptr/core/grammar/subjunctive.py

"""
Subjunctive Mood Handler
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SubjunctiveType(Enum):
    PRESENT = "present"      # I suggest that he go
    PAST = "past"            # If I were rich
    PAST_PERFECT = "past_perfect"  # If only I had known
    MANDATIVE = "mandative"  # It is necessary that she be here


@dataclass
class SubjunctiveInfo:
    is_subjunctive: bool
    type: Optional[SubjunctiveType]
    main_verb: Optional[str]
    that_clause: Optional[str]
    trigger_word: Optional[str]  # suggest, recommend, necessary, etc.
    meaning: str = ""


class SubjunctiveHandler:
    """
    Handles all subjunctive mood constructions
    """
    
    def __init__(self):
        # ============================================================
        # Verbs that trigger present subjunctive (suggest/recommend/insist that + base verb)
        # ============================================================
        self.present_subjunctive_verbs = {
            "suggest": "အကြံပြုသည်",
            "recommend": "အကြံပေးသည်",
            "insist": "အခိုင်အမာပြောဆိုသည်",
            "demand": "တောင်းဆိုသည်",
            "request": "တောင်းပန်သည်",
            "propose": "အဆိုပြုသည်",
            "move": "အဆိုတင်သွင်းသည်",
            "urge": "တိုက်တွန်းသည်",
            "ask": "တောင်းဆိုသည်",
            "require": "လိုအပ်သည်",
            "advise": "အကြံပေးသည်"
        }
        
        # ============================================================
        # Adjectives that trigger mandative subjunctive (it is + adj + that + base verb)
        # ============================================================
        self.mandative_adjectives = {
            "necessary": "လိုအပ်သော",
            "important": "အရေးကြီးသော",
            "essential": "မရှိမဖြစ်",
            "vital": "အသက်သွေးကြော",
            "crucial": "အရေးပါသော",
            "advisable": "သင့်လျော်သော",
            "preferable": "ပိုနှစ်သက်ဖွယ်",
            "recommended": "အကြံပြုထားသော",
            "requested": "တောင်းဆိုထားသော",
            "required": "လိုအပ်သော",
            "mandatory": "မဖြစ်မနေ"
        }
        
        # ============================================================
        # Expressions for past subjunctive (wish, if only, as if, as though)
        # ============================================================
        self.past_subjunctive_expressions = {
            "wish": "ဆန္ဒရှိသည် (ဖြစ်ချင်သော်လည်းမဖြစ်)",
            "if only": "ဖြစ်ချင်သည် (တကယ်မဟုတ်)",
            "as if": "သကဲ့သို့",
            "as though": "သကဲ့သို့"
        }
        
        # ============================================================
        # Past perfect subjunctive expressions (wish, if only about past)
        # ============================================================
        self.past_perfect_expressions = {
            "wish": "တမင်တကာ ဆန္ဒရှိသည် (အတိတ်မှာ မဖြစ်ခဲ့)",
            "if only": "ဖြစ်ခဲ့ပါက (အတိတ်မှာ မဖြစ်ခဲ့)"
        }
    
    def detect_subjunctive(self, tokens: List[str]) -> SubjunctiveInfo:
        """
        Detect subjunctive mood in sentence
        """
        sentence = " ".join(tokens).lower()
        
        # ============================================================
        # 1. Present subjunctive (that + base verb)
        # Pattern: [verb] + that + [subject] + [base verb]
        # ============================================================
        for verb, meaning in self.present_subjunctive_verbs.items():
            if verb in sentence and "that" in sentence:
                # Find the clause after "that"
                parts = sentence.split("that", 1)
                if len(parts) == 2:
                    that_clause = parts[1].strip()
                    # Check if verb is in base form (not 3rd person singular)
                    if not that_clause.split()[1].endswith("s") if len(that_clause.split()) > 1 else False:
                        return SubjunctiveInfo(
                            is_subjunctive=True,
                            type=SubjunctiveType.PRESENT,
                            main_verb=verb,
                            that_clause=that_clause,
                            trigger_word=verb,
                            meaning=f"{meaning} (ကြိယာပုံစံမပြောင်း)"
                        )
        
        # ============================================================
        # 2. Mandative subjunctive (It is + adj + that + base verb)
        # ============================================================
        for adj, meaning in self.mandative_adjectives.items():
            if f"it is {adj}" in sentence or f"it was {adj}" in sentence:
                if "that" in sentence:
                    parts = sentence.split("that", 1)
                    if len(parts) == 2:
                        that_clause = parts[1].strip()
                        return SubjunctiveInfo(
                            is_subjunctive=True,
                            type=SubjunctiveType.MANDATIVE,
                            main_verb=None,
                            that_clause=that_clause,
                            trigger_word=adj,
                            meaning=f"{meaning} (မဖြစ်မနေ + base verb)"
                        )
        
        # ============================================================
        # 3. Past subjunctive (wish, if only + were/past tense)
        # ============================================================
        for expr, meaning in self.past_subjunctive_expressions.items():
            if expr in sentence:
                # Check for "were" (special subjunctive for all persons)
                if "were" in sentence:
                    # Extract the clause
                    clause = sentence[sentence.find(expr) + len(expr):].strip()
                    return SubjunctiveInfo(
                        is_subjunctive=True,
                        type=SubjunctiveType.PAST,
                        main_verb=expr,
                        that_clause=clause,
                        trigger_word=expr,
                        meaning=f"{meaning} (were ဖြင့်သုံးသည်)"
                    )
                # Check for past tense verb
                elif any(word.endswith("ed") for word in tokens):
                    return SubjunctiveInfo(
                        is_subjunctive=True,
                        type=SubjunctiveType.PAST,
                        main_verb=expr,
                        that_clause="",
                        trigger_word=expr,
                        meaning=f"{meaning} (လက်ရှိမဖြစ်သောအရာ)"
                    )
        
        # ============================================================
        # 4. Past perfect subjunctive (wish, if only + had + past participle)
        # ============================================================
        for expr, meaning in self.past_perfect_expressions.items():
            if expr in sentence and "had" in sentence:
                return SubjunctiveInfo(
                    is_subjunctive=True,
                    type=SubjunctiveType.PAST_PERFECT,
                    main_verb=expr,
                    that_clause="",
                    trigger_word=expr,
                    meaning=f"{meaning} (အတိတ်မှာမဖြစ်ခဲ့)"
                )
        
        return SubjunctiveInfo(False, None, None, None, None)
    
    def get_subjunctive_rules(self) -> Dict:
        """Get subjunctive usage rules"""
        return {
            "present_subjunctive": {
                "ဖွဲ့စည်းပုံ": "Verb(suggest/recommend) + that + subject + base verb (no -s)",
                "example": "I suggest that he go to the doctor",
                "myanmar": "သူဆရာဝန်သွားရန် အကြံပြုသည်"
            },
            "mandative_subjunctive": {
                "ဖွဲ့စည်းပုံ": "It is + adjective + that + subject + base verb",
                "example": "It is necessary that she be here on time",
                "myanmar": "သူမအချိန်မီရောက်ရန် လိုအပ်သည်"
            },
            "past_subjunctive": {
                "ဖွဲ့စည်းပုံ": "wish/as if + subject + were (or past tense)",
                "example": "I wish I were rich",
                "myanmar": "ငါချမ်းသာချင်သည် (မချမ်းသာပါ)"
            },
            "past_perfect_subjunctive": {
                "ဖွဲ့စည်းပုံ": "wish/if only + subject + had + past participle",
                "example": "If only I had known the answer",
                "myanmar": "ငါအဖြေသိခဲ့ရင် (မသိခဲ့ပါ)"
            }
        }
