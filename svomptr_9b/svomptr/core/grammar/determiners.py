# /svomptr_9b/svomptr/core/grammar/determiners.py

"""
Complete Determiners Handler
Articles, Demonstratives, Possessives, Quantifiers
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DeterminerType(Enum):
    ARTICLE = "article"              # a, an, the
    DEMONSTRATIVE = "demonstrative"  # this, that, these, those
    POSSESSIVE = "possessive"        # my, your, his, her, its, our, their
    QUANTIFIER = "quantifier"        # some, any, many, much, few, little, all, both, each, every
    NUMERAL = "numeral"              # one, two, three, first, second


class ArticleType(Enum):
    DEFINITE = "definite"     # the
    INDEFINITE = "indefinite" # a, an
    ZERO = "zero"             # no article


@dataclass
class DeterminerInfo:
    """Information about a determiner"""
    type: DeterminerType
    word: str
    modifies: str  # The noun it modifies
    position: int
    meaning: str = ""
    article_type: Optional[ArticleType] = None
    is_countable: Optional[bool] = None


class DeterminerHandler:
    """
    Handles ALL determiners (words that come before nouns)
    Complete implementation
    """
    
    def __init__(self):
        # ============================================================
        # ARTICLES
        # ============================================================
        self.articles = {
            ArticleType.DEFINITE: ["the"],
            ArticleType.INDEFINITE: ["a", "an"],
            ArticleType.ZERO: []
        }
        
        # Article meanings in Myanmar
        self.article_meanings = {
            "the": "ထို/အဆိုပါ (သတ်မှတ်ထားသော)",
            "a": "တစ်ခုသော (ယေဘုယျ)",
            "an": "တစ်ခုသော (ယေဘုယျ - သရအက္ခရာရှေ့)"
        }
        
        # Vowel sounds for 'an' detection
        self.vowel_sounds = ['a', 'e', 'i', 'o', 'u', 'အ', 'ဤ', 'ဥ', 'ဧ', 'ဩ']
        
        # ============================================================
        # DEMONSTRATIVES
        # ============================================================
        self.demonstratives = {
            "near_singular": ["this"],
            "near_plural": ["these"],
            "far_singular": ["that"],
            "far_plural": ["those"]
        }
        
        self.demonstrative_meanings = {
            "this": "ဒီ/ဤ (အနီးအနား - တစ်ခု)",
            "these": "ဒီဟာတွေ/ဤအရာများ (အနီးအနား - အများ)",
            "that": "ဟို/ထို (အဝေးတစ်နေရာ - တစ်ခု)",
            "those": "ဟိုဟာတွေ/ထိုအရာများ (အဝေးတစ်နေရာ - အများ)"
        }
        
        # ============================================================
        # POSSESSIVES
        # ============================================================
        self.possessive_adjectives = {
            "1st_singular": {"word": "my", "meaning": "ကျွန်တော်၏/ကျွန်မ၏"},
            "2nd_singular": {"word": "your", "meaning": "ခင်ဗျား၏/ရှင်၏"},
            "3rd_masculine": {"word": "his", "meaning": "သူ၏ (ယောက်ျား)"},
            "3rd_feminine": {"word": "her", "meaning": "သူမ၏ (မိန်းမ)"},
            "3rd_neuter": {"word": "its", "meaning": "၎င်း၏ (အရာ/တိရစ္ဆာန်)"},
            "1st_plural": {"word": "our", "meaning": "ကျွန်တော်တို့၏"},
            "2nd_plural": {"word": "your", "meaning": "ခင်ဗျားတို့၏"},
            "3rd_plural": {"word": "their", "meaning": "သူတို့၏"},
            "interrogative": {"word": "whose", "meaning": "မည်သူ၏"}
        }
        
        # Possessive pronouns (stand alone)
        self.possessive_pronouns = {
            "mine": "ငါ့ဟာ",
            "yours": "မင်းဟာ",
            "his": "သူ့ဟာ",
            "hers": "သူမရဲ့ဟာ",
            "its": "၎င်းရဲ့ဟာ",
            "ours": "ငါတို့ရဲ့ဟာ",
            "theirs": "သူတို့ရဲ့ဟာ"
        }
        
        # ============================================================
        # QUANTIFIERS - Complete list
        # ============================================================
        self.quantifiers = {
            # Positive large quantity
            "positive_large_countable": ["many", "numerous", "countless", "a great many"],
            "positive_large_uncountable": ["much", "a great deal of", "an abundance of"],
            
            # Positive medium quantity
            "positive_medium_countable": ["several", "a number of", "quite a few"],
            "positive_medium_uncountable": ["some", "a bit of", "an amount of"],
            
            # Positive small quantity
            "positive_small_countable": ["a few", "few", "a couple of"],
            "positive_small_uncountable": ["a little", "little"],
            
            # Universal/everything
            "universal_all": ["all", "every", "each", "every single"],
            "universal_both": ["both"],
            "universal_either_neither": ["either", "neither"],
            
            # Negative/zero
            "negative_zero": ["no", "none", "not any"],
            
            # Question/condition
            "question": ["any", "some"],
            
            # Sufficiency
            "sufficiency": ["enough", "sufficient", "plenty of"],
            
            # Excess
            "excess": ["too many", "too much", "excessive"],
            
            # Indefinite
            "indefinite": ["any", "some", "certain", "various", "different"]
        }
        
        # Quantifier meanings in Myanmar
        self.quantifier_meanings = {
            "many": "များစွာသော (ရေတွက်နိုင်သော)",
            "much": "များစွာသော (ရေတွက်မရသော)",
            "some": "အချို့သော",
            "any": "မည်သည့်",
            "a few": "အနည်းငယ်သော (ရေတွက်နိုင်)",
            "a little": "အနည်းငယ်သော (ရေတွက်မရ)",
            "few": "အနည်းငယ်မျှသော (ရေတွက်နိုင်)",
            "little": "အနည်းငယ်မျှသော (ရေတွက်မရ)",
            "all": "အားလုံးသော",
            "every": "တိုင်းသော",
            "each": "တစ်ခုချင်းစီသော",
            "both": "နှစ်ခုလုံး",
            "either": "တစ်ခုခု",
            "neither": "တစ်ခုမှမဟုတ်",
            "no": "မရှိ",
            "none": "တစ်ခုမှမရှိ",
            "enough": "လုံလောက်သော",
            "several": "အတော်များများသော",
            "numerous": "မြောက်မြားစွာသော"
        }
        
        # Countable vs Uncountable nouns (common)
        self.countable_nouns = ["cat", "dog", "car", "book", "person", "apple", "chair", "table"]
        self.uncountable_nouns = ["water", "rice", "air", "information", "knowledge", "money", "time", "weather"]
    
    def detect_determiners(self, tokens: List[str]) -> List[DeterminerInfo]:
        """Detect ALL determiners in sentence"""
        determiners = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            # ========================================================
            # 1. ARTICLES
            # ========================================================
            if token_lower in self.articles[ArticleType.DEFINITE]:
                determiners.append(DeterminerInfo(
                    type=DeterminerType.ARTICLE,
                    word=token,
                    modifies=tokens[i+1] if i+1 < len(tokens) else "",
                    position=i,
                    meaning=self.article_meanings.get(token_lower, ""),
                    article_type=ArticleType.DEFINITE
                ))
            
            elif token_lower in self.articles[ArticleType.INDEFINITE]:
                # Determine if 'an' is used correctly
                article_type = ArticleType.INDEFINITE
                determiners.append(DeterminerInfo(
                    type=DeterminerType.ARTICLE,
                    word=token,
                    modifies=tokens[i+1] if i+1 < len(tokens) else "",
                    position=i,
                    meaning=self.article_meanings.get(token_lower, ""),
                    article_type=article_type
                ))
            
            # ========================================================
            # 2. DEMONSTRATIVES
            # ========================================================
            for category, words in self.demonstratives.items():
                if token_lower in words:
                    determiners.append(DeterminerInfo(
                        type=DeterminerType.DEMONSTRATIVE,
                        word=token,
                        modifies=tokens[i+1] if i+1 < len(tokens) else "",
                        position=i,
                        meaning=self.demonstrative_meanings.get(token_lower, "")
                    ))
                    break
            
            # ========================================================
            # 3. POSSESSIVE ADJECTIVES
            # ========================================================
            for key, info in self.possessive_adjectives.items():
                if token_lower == info["word"]:
                    determiners.append(DeterminerInfo(
                        type=DeterminerType.POSSESSIVE,
                        word=token,
                        modifies=tokens[i+1] if i+1 < len(tokens) else "",
                        position=i,
                        meaning=info["meaning"]
                    ))
                    break
            
            # ========================================================
            # 4. QUANTIFIERS
            # ========================================================
            for category, words in self.quantifiers.items():
                if token_lower in words:
                    # Check if noun is countable or uncountable
                    is_countable = None
                    if i+1 < len(tokens):
                        next_word = tokens[i+1].lower()
                        if next_word in self.countable_nouns:
                            is_countable = True
                        elif next_word in self.uncountable_nouns:
                            is_countable = False
                    
                    determiners.append(DeterminerInfo(
                        type=DeterminerType.QUANTIFIER,
                        word=token,
                        modifies=tokens[i+1] if i+1 < len(tokens) else "",
                        position=i,
                        meaning=self.quantifier_meanings.get(token_lower, ""),
                        is_countable=is_countable
                    ))
                    break
        
        return determiners
    
    def get_article_rules(self) -> Dict:
        """Get article usage rules"""
        return {
            "definite_article": {
                "သုံးရမည့်အချိန်": [
                    "အရင်ဖော်ပြပြီးသားအရာ (Previously mentioned)",
                    "တစ်ခုတည်းသောအရာ (Unique things - the sun, the moon)",
                    "သိပြီးသားအရာ (Known to both speaker and listener)",
                    "နာမဝိသေသနအဆင့်မြင့် (Superlatives - the best)",
                    "အခန်းဆက်များ (Ordinal numbers - the first)"
                ]
            },
            "indefinite_article": {
                "သုံးရမည့်အချိန်": [
                    "ပထမဆုံးမိတ်ဆက်ချိန် (First mention)",
                    "ယေဘုယျအားဖြင့် (In general)",
                    "အလုပ်အကိုင်ဖော်ပြချိန် (Professions - a doctor)",
                    "နံပါတ်တစ်ကိုပြချိန် (One - a hundred)"
                ]
            },
            "zero_article": {
                "သုံးရမည့်အချိန်": [
                    "အများကိန်းယေဘုယျ (Plural general - Cats are cute)",
                    "ရေတွက်မရသောနာမ် (Uncountable - Water is life)",
                    "ဘာသာစကားများ (Languages - English)",
                    "အားကစားများ (Sports - football)",
                    "နိုင်ငံများ (Countries - Myanmar)"
                ]
            }
        }
    
    def get_demonstrative_rules(self) -> Dict:
        """Get demonstrative usage rules"""
        return {
            "this/these": {
                "meaning": "အနီးအနား (Near in distance or time)",
                "examples": ["This book (ဒီစာအုပ်)", "These days (ဒီနေ့ရက်များ)"]
            },
            "that/those": {
                "meaning": "အဝေးတစ်နေရာ (Far in distance or time)",
                "examples": ["That building (ဟိုအဆောက်အအုံ)", "Those days (ဟိုနေ့ရက်များ)"]
            }
        }
    
    def get_possessive_rules(self) -> Dict:
        """Get possessive usage rules"""
        return {
            "possessive_adjectives": {
                "usage": "နာမ်ရှေ့မှာ ထားရသည် (Before noun)",
                "examples": ["my car", "her house", "their children"]
            },
            "possessive_pronouns": {
                "usage": "နာမ်နေရာမှာ သုံးသည် (Replace noun)",
                "examples": ["This is mine", "The book is yours"]
            }
        }
    
    def get_quantifier_rules(self) -> Dict:
        """Get quantifier usage rules"""
        return {
            "countable": {
                "many": "များစွာ",
                "a few": "အနည်းငယ် (ရှိသေး)",
                "few": "အနည်းငယ် (မရှိသလောက်)",
                "several": "အတော်များများ"
            },
            "uncountable": {
                "much": "များစွာ",
                "a little": "အနည်းငယ် (ရှိသေး)",
                "little": "အနည်းငယ် (မရှိသလောက်)"
            },
            "both": {
                "some": "အချို့",
                "any": "မည်သည့် (negative/question)",
                "no": "မရှိ",
                "all": "အားလုံး",
                "every": "တိုင်း"
            }
        }
    
    def get_determiner_summary(self, determiners: List[DeterminerInfo]) -> str:
        """Get human-readable summary of determiners"""
        if not determiners:
            return "No determiners found"
        
        lines = []
        for d in determiners:
            lines.append(f"  {d.type.value}: '{d.word}' → modifies '{d.modifies}'")
            if d.meaning:
                lines.append(f"    Meaning: {d.meaning}")
            if d.article_type:
                lines.append(f"    Article type: {d.article_type.value}")
            if d.is_countable is not None:
                lines.append(f"    Countable: {d.is_countable}")
        
        return "\n".join(lines)
