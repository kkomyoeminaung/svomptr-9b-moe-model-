# /svomptr_9b/svomptr/core/grammar/adverbs.py

"""
Enhanced Adverbs Handler
Adverbs of degree, frequency, manner, place, time
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AdverbType(Enum):
    DEGREE = "degree"          # very, quite, too, so, enough, almost, nearly, hardly
    FREQUENCY = "frequency"    # always, often, sometimes, rarely, never, every day
    MANNER = "manner"          # quickly, slowly, carefully, beautifully
    PLACE = "place"            # here, there, everywhere, inside, outside
    TIME = "time"              # now, then, today, yesterday, tomorrow
    FOCUS = "focus"            # only, even, just, also, too
    VIEWPOINT = "viewpoint"    # frankly, honestly, generally, technically


@dataclass
class AdverbInfo:
    type: AdverbType
    word: str
    meaning: str
    position: int
    modifies: str  # verb, adjective, or whole sentence
    intensity: float = 0.5  # 0-1 scale
    frequency_pct: Optional[float] = None


class AdverbsHandler:
    """
    Handles ALL adverb types
    """
    
    def __init__(self):
        # ============================================================
        # 1. Adverbs of Degree (အတိုင်းအတာပြကြိယာဝိသေသန)
        # ============================================================
        self.degree_adverbs = {
            "very": {"intensity": 0.9, "meaning": "အလွန်"},
            "quite": {"intensity": 0.7, "meaning": "အတော်"},
            "rather": {"intensity": 0.6, "meaning": "အတန်အသင့်"},
            "too": {"intensity": 0.95, "meaning": "လွန်းသည်"},
            "so": {"intensity": 0.85, "meaning": "အလွန်အမင်း"},
            "enough": {"intensity": 0.5, "meaning": "လုံလောက်သော"},
            "almost": {"intensity": 0.9, "meaning": "နီးပါး"},
            "nearly": {"intensity": 0.85, "meaning": "နီးပါး"},
            "hardly": {"intensity": 0.1, "meaning": "တစ်စိတ်တစ်ဒေသမျှ"},
            "barely": {"intensity": 0.1, "meaning": "မဲ့ကန်အဲ"},
            "completely": {"intensity": 1.0, "meaning": "လုံးဝ"},
            "totally": {"intensity": 1.0, "meaning": "လုံးဝ"},
            "extremely": {"intensity": 1.0, "meaning": "အလွန်အမင်း"},
            "slightly": {"intensity": 0.2, "meaning": "အနည်းငယ်"},
            "somewhat": {"intensity": 0.4, "meaning": "အလယ်အလတ်"}
        }
        
        # ============================================================
        # 2. Adverbs of Frequency (အကြိမ်ရေပြကြိယာဝိသေသန)
        # ============================================================
        self.frequency_adverbs = {
            "always": {"pct": 1.0, "meaning": "အမြဲတမ်း"},
            "constantly": {"pct": 0.95, "meaning": "မပြတ်"},
            "usually": {"pct": 0.9, "meaning": "များသောအားဖြင့်"},
            "often": {"pct": 0.7, "meaning": "မကြာခဏ"},
            "frequently": {"pct": 0.7, "meaning": "မကြာခဏ"},
            "sometimes": {"pct": 0.5, "meaning": "တစ်ခါတစ်ရံ"},
            "occasionally": {"pct": 0.3, "meaning": "ရံဖန်ရံခါ"},
            "rarely": {"pct": 0.1, "meaning": "ခဲခဲယင်းယင်း"},
            "seldom": {"pct": 0.1, "meaning": "ရှားရှားပါးပါး"},
            "hardly ever": {"pct": 0.05, "meaning": "မဖြစ်သလောက်"},
            "never": {"pct": 0.0, "meaning": "လုံးဝမရှိ"},
            "every day": {"pct": 1.0, "meaning": "နေ့တိုင်း"},
            "weekly": {"pct": 0.14, "meaning": "အပတ်စဉ်"},
            "monthly": {"pct": 0.03, "meaning": "လစဉ်"},
            "yearly": {"pct": 0.003, "meaning": "နှစ်စဉ်"}
        }
        
        # ============================================================
        # 3. Adverbs of Manner (ပုံစံပြကြိယာဝိသေသန)
        # ============================================================
        self.manner_adverbs = {
            "quickly": "လျင်မြန်စွာ",
            "slowly": "ဖြည်းညှင်းစွာ",
            "carefully": "ဂရုတစိုက်",
            "carelessly": "ဂရုမစိုက်",
            "beautifully": "လှပစွာ",
            "badly": "ဆိုးဆိုးဝါးဝါး",
            "well": "ကောင်းစွာ",
            "easily": "လွယ်ကူစွာ",
            "difficultly": "ခက်ခဲစွာ",
            "happily": "ပျော်ရွှင်စွာ",
            "sadly": "ဝမ်းနည်းစွာ",
            "angrily": "ဒေါသထွက်စွာ",
            "politely": "ယဉ်ကျေးစွာ",
            "rudely": "ရိုင်းစိုင်းစွာ"
        }
        
        # ============================================================
        # 4. Adverbs of Place (နေရာပြကြိယာဝိသေသန)
        # ============================================================
        self.place_adverbs = {
            "here": "ဒီမှာ",
            "there": "ဟိုမှာ",
            "everywhere": "နေရာတိုင်း",
            "anywhere": "ဘယ်နေရာမှာ",
            "nowhere": "ဘယ်နေရာမှမရှိ",
            "inside": "အတွင်းဘက်",
            "outside": "အပြင်ဘက်",
            "upstairs": "အပေါ်ထပ်",
            "downstairs": "အောက်ထပ်",
            "abroad": "နိုင်ငံရပ်ခြား",
            "home": "အိမ်",
            "away": "အဝေးကို",
            "near": "အနီး",
            "far": "အဝေး"
        }
        
        # ============================================================
        # 5. Adverbs of Time (အချိန်ပြကြိယာဝိသေသန) - Enhanced
        # ============================================================
        self.time_adverbs = {
            "now": "အခု",
            "then": "ထိုအချိန်က",
            "today": "ဒီနေ့",
            "yesterday": "မနေ့က",
            "tomorrow": "မနက်ဖြန်",
            "tonight": "ဒီည",
            "soon": "မကြာမီ",
            "later": "နောက်မှ",
            "early": "စောစော",
            "late": "နောက်ကျ",
            "immediately": "ချက်ချင်း",
            "recently": "မကြာသေးမီက",
            "already": "ပြီးပြီ",
            "yet": "သေး",
            "still": "သေးသည်",
            "just": "ပဲ",
            "ago": "လွန်ခဲ့သော"
        }
        
        # ============================================================
        # 6. Focus Adverbs (အာရုံစိုက်ပြကြိယာဝိသေသန)
        # ============================================================
        self.focus_adverbs = {
            "only": "သာ",
            "even": "ပင်",
            "just": "မျှ",
            "alone": "တစ်ခုတည်းသော",
            "particularly": "အထူးသဖြင့်",
            "especially": "အထူးသဖြင့်",
            "specifically": "အတိအကျ",
            "exactly": "အတိအကျ",
            "simply": "ရိုးရိုးရှင်းရှင်း"
        }
        
        # ============================================================
        # 7. Viewpoint Adverbs (အမြင်ပြကြိယာဝိသေသန)
        # ============================================================
        self.viewpoint_adverbs = {
            "frankly": "ရိုးရိုးသားသားပြောရရင်",
            "honestly": "ရိုးသားစွာ",
            "generally": "ယေဘုယျအားဖြင့်",
            "technically": "နည်းပညာအရ",
            "theoretically": "သီအိုရီအရ",
            "practically": "လက်တွေ့အရ",
            "personally": "ကိုယ်ပိုင်အမြင်",
            "seriously": "လေးလေးနက်နက်",
            "strictly": "တင်းတင်းကျပ်ကျပ်"
        }
    
    def detect_adverbs(self, tokens: List[str]) -> List[AdverbInfo]:
        """Detect ALL adverbs in sentence"""
        adverbs = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            # Degree adverbs
            if token_lower in self.degree_adverbs:
                info = self.degree_adverbs[token_lower]
                adverbs.append(AdverbInfo(
                    type=AdverbType.DEGREE,
                    word=token,
                    meaning=info["meaning"],
                    position=i,
                    modifies="adj/adv",
                    intensity=info["intensity"]
                ))
            
            # Frequency adverbs
            elif token_lower in self.frequency_adverbs:
                info = self.frequency_adverbs[token_lower]
                adverbs.append(AdverbInfo(
                    type=AdverbType.FREQUENCY,
                    word=token,
                    meaning=info["meaning"],
                    position=i,
                    modifies="verb",
                    frequency_pct=info["pct"]
                ))
            
            # Manner adverbs
            elif token_lower in self.manner_adverbs:
                adverbs.append(AdverbInfo(
                    type=AdverbType.MANNER,
                    word=token,
                    meaning=self.manner_adverbs[token_lower],
                    position=i,
                    modifies="verb"
                ))
            
            # Place adverbs
            elif token_lower in self.place_adverbs:
                adverbs.append(AdverbInfo(
                    type=AdverbType.PLACE,
                    word=token,
                    meaning=self.place_adverbs[token_lower],
                    position=i,
                    modifies="verb"
                ))
            
            # Time adverbs
            elif token_lower in self.time_adverbs:
                adverbs.append(AdverbInfo(
                    type=AdverbType.TIME,
                    word=token,
                    meaning=self.time_adverbs[token_lower],
                    position=i,
                    modifies="verb/whole"
                ))
            
            # Focus adverbs
            elif token_lower in self.focus_adverbs:
                adverbs.append(AdverbInfo(
                    type=AdverbType.FOCUS,
                    word=token,
                    meaning=self.focus_adverbs[token_lower],
                    position=i,
                    modifies="noun/verb"
                ))
            
            # Viewpoint adverbs
            elif token_lower in self.viewpoint_adverbs:
                adverbs.append(AdverbInfo(
                    type=AdverbType.VIEWPOINT,
                    word=token,
                    meaning=self.viewpoint_adverbs[token_lower],
                    position=i,
                    modifies="whole sentence"
                ))
        
        return adverbs
    
    def get_adverb_rules(self) -> Dict:
        """Get adverb usage rules"""
        return {
            "degree": {
                "description": "အတိုင်းအတာပြရန်",
                "position": "adjective/adverb ရှေ့မှာ",
                "example": "very good, too slow"
            },
            "frequency": {
                "description": "အကြိမ်ရေပြရန်",
                "position": "main verb ရှေ့ (be verb နောက်)",
                "example": "always happy, never sleeps"
            },
            "manner": {
                "description": "ပုံစံပြရန် (ဘယ်လို)",
                "position": "verb နောက်",
                "example": "run quickly, speak softly"
            },
            "place": {
                "description": "နေရာပြရန် (ဘယ်မှာ)",
                "position": "verb နောက် (သို့) sentence အဆုံး",
                "example": "sit here, go outside"
            },
            "time": {
                "description": "အချိန်ပြရန် (ဘယ်တော့)",
                "position": "sentence အစ သို့ အဆုံး",
                "example": "today we go, we go today"
            },
            "focus": {
                "description": "အာရုံစိုက်ရန်",
                "position": "အာရုံစိုက်လိုသော word ရှေ့",
                "example": "only you, just one"
            },
            "viewpoint": {
                "description": "အမြင်ပြရန်",
                "position": "sentence အစ (comma ခံ)",
                "example": "Frankly, I don't care"
            }
        }
    
    def get_adverb_summary(self, adverbs: List[AdverbInfo]) -> str:
        """Get human-readable summary of adverbs"""
        if not adverbs:
            return "No adverbs found"
        
        lines = []
        for a in adverbs:
            line = f"  {a.type.value}: '{a.word}' → {a.meaning}"
            if a.intensity:
                line += f" (intensity: {a.intensity})"
            if a.frequency_pct:
                line += f" (frequency: {a.frequency_pct*100}%)"
            lines.append(line)
        
        return "\n".join(lines)
