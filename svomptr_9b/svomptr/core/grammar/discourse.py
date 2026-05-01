# /svomptr_9b/svomptr/core/grammar/discourse.py

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DiscourseInfo:
    has_marker: bool
    markers: List[str]
    positions: List[int]
    meaning: str


class DiscourseHandler:
    """
    Handles discourse markers (linking words for conversation flow)
    Based on Myo Min Aung's specification
    """
    
    def __init__(self):
        # Discourse markers categorized by function
        self.discourse_markers = {
            "sequencing": {
                "words": ["first", "second", "third", "next", "then", "finally", "lastly",
                          "ပထမ", "ဒုတိယ", "နောက်", "နောက်ဆုံး"],
                "meaning": "အစီအစဉ်ပြရန်"
            },
            "addition": {
                "words": ["also", "too", "moreover", "furthermore", "in addition", "besides",
                          "ထို့အပြင်", "နောက်ပြီး"],
                "meaning": "ထပ်ထည့်ရန်"
            },
            "contrast": {
                "words": ["however", "nevertheless", "nonetheless", "on the other hand", "in contrast",
                          "သို့ပေမယ့်", "သို့ရာတွင်"],
                "meaning": "ဆန့်ကျင်ဖော်ပြရန်"
            },
            "cause_effect": {
                "words": ["therefore", "thus", "hence", "consequently", "as a result",
                          "ထို့ကြောင့်", "ဒါကြောင့်"],
                "meaning": "အကျိုးဆက်ပြရန်"
            },
            "clarification": {
                "words": ["specifically", "namely", "in other words", "that is", "i.e.",
                          "ဆိုလိုသည်မှာ", "အတိအကျဆိုရင်"],
                "meaning": "ရှင်းလင်းချက်ထည့်ရန်"
            },
            "concession": {
                "words": ["admittedly", "of course", "to be sure", "it is true that",
                          "ဝန်ခံရလျှင်", "ဧကန်စင်စစ်"],
                "meaning": "ဝန်ခံချက်ပြုရန်"
            },
            "summary": {
                "words": ["in conclusion", "to summarize", "in short", "overall", "all in all",
                          "နိဂုံးချုပ်ရလျှင်", "ခြုံငုံသုံးသပ်ရလျှင်"],
                "meaning": "အနှစ်ချုပ်ရန်"
            },
            "filler": {
                "words": ["well", "actually", "basically", "honestly", "frankly", "to be honest",
                          "ဟုတ်တယ်", "တကယ်တော့", "ရိုးရိုးသားသားပြောရရင်"],
                "meaning": "စကားစပြီး ချောမွေ့အောင်"
            }
        }
        
        # Sentence-initial discourse markers (most common position)
        self.sentence_initial = ["well", "actually", "basically", "honestly", "frankly",
                                 "however", "therefore", "thus", "consequently",
                                 "ဟုတ်တယ်", "တကယ်တော့", "ထို့ကြောင့်"]
    
    def detect_discourse_markers(self, tokens: List[str]) -> DiscourseInfo:
        """
        Detect discourse markers in sentence
        """
        markers_found = []
        positions = []
        meaning = ""
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            for category, info in self.discourse_markers.items():
                if token_lower in info["words"]:
                    markers_found.append(token)
                    positions.append(i)
                    meaning = info["meaning"]
                    break
        
        return DiscourseInfo(
            has_marker=len(markers_found) > 0,
            markers=markers_found,
            positions=positions,
            meaning=meaning
        )
    
    def is_sentence_initial_marker(self, token: str) -> bool:
        """Check if token is a sentence-initial discourse marker"""
        return token.lower() in self.sentence_initial
    
    def remove_markers(self, tokens: List[str]) -> List[str]:
        """Remove discourse markers for core parsing"""
        info = self.detect_discourse_markers(tokens)
        if not info.has_marker:
            return tokens
        
        # Remove markers from the list
        result = []
        for i, token in enumerate(tokens):
            if i not in info.positions:
                result.append(token)
        
        return result
