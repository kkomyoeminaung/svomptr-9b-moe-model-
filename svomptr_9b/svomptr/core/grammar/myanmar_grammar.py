# /svomptr_9b/svomptr/core/grammar/myanmar_grammar.py

"""
Myanmar Grammar Handler - Complete Module
Verb Suffixes, Tense Markers, Honorifics, Particles, Syntax
Based on Myo Min Aung's specification
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MyanmarTense(Enum):
    PRESENT = "present"      # သည်၊ နေသည်
    PAST = "past"            # ခဲ့သည်
    FUTURE = "future"        # မည်၊ လိမ့်မည်
    NEGATIVE = "negative"    # မ...ဘူး


class PolitenessLevel(Enum):
    HIGH = "high"      # ပါ၊ ခင်ဗျာ၊ ရှင်
    MEDIUM = "medium"  # Standard (no special markers)
    LOW = "low"        # ကွ၊ ဟေ့ (informal)


@dataclass
class MyanmarVerbInfo:
    root: str
    suffix: str
    tense: MyanmarTense
    is_negative: bool = False
    politeness: PolitenessLevel = PolitenessLevel.MEDIUM


@dataclass
class MyanmarSlotMapping:
    original_text: str
    slot: str
    particle: str
    core_meaning: str
    position: int


class MyanmarGrammarHandler:
    """
    Complete Myanmar Grammar Handler for SVOMPTR
    Handles: Particles, Tense markers, Honorifics, Syntax
    """
    
    def __init__(self):
        # ============================================================
        # 1. Particles (ဝိဘတ်များ) by SVOMPTR Slot
        # ============================================================
        self.particles = {
            # Subject markers (ကတ္တားပစ္စည်း) - သည် (Lit), က (Coll), မှာ
            "S": ["သည်", "က", "မှာ", "ဦးဇင်း", "ကျွန်တော်မျိုး"],
            
            # Object markers (ကံပစ္စည်း) - ကို (Direct), အား (Indirect/Polite)
            "O": ["ကို", "အား", "ကိုတော့", "ထံသို့"],
            
            # Manner/Emotion markers (လုပ်နည်းပြဝိဘတ်) - ဖြင့်, နှင့်, စွာ
            "M": ["ဖြင့်", "နှင့်", "စွာ", "လျက်", "လျောင်း", "သဖြင့်", "တုန်းက"],
            
            # Place markers (ဌာနိယပစ္စည်း) - ၌, တွင်, မှာ, ဝယ်
            "P": ["၌", "တွင်", "မှာ", "ဝယ်", "အတွင်း", "အပြင်", "ထက်", "အောက်", "ဘက်", "ဆီသို့"],
            
            # Time markers (ကာလပစ္စည်း) - ကတည်းက, အချိန်
            "T": ["ကတည်းက", "အချိန်", "အခါ", "မှာ", "ခေတ်", "အထိ", "လောက်", "အတွင်းမှာ", "တိုင်း"],
            
            # Reason markers (အကြောင်းပြဝိဘတ်) - ကြောင့်, သောကြောင့်
            "R": ["ကြောင့်", "သောကြောင့်", "မို့လို့", "အတွက်", "ဖြင့်", "ရာတွင်", "၍"]
        }

        # New: Possessive Markers
        self.possessive_particles = ["၏", "ရဲ့", "တို့၏", "တို့ရဲ့"]

        # New: Numeral Classifiers (သင်္ချာပစ္စည်း)
        self.classifiers = {
            "people": ["ယောက်", "ပါး", "ဦး"],
            "animals": ["ကောင်"],
            "objects": ["ခု", "အုပ်", "စင်း", "ထည်", "လုံး", "ချောင်း", "ပင်"],
            "abstract": ["ချက်", "ခု", "မျိုး"]
        }

        # New: Auxiliary Verbs (ကြိယာကူများ)
        self.aux_verbs = {
            "desire": ["ချင်", "လို"],
            "ability": ["နိုင်", "တတ်"],
            "permission/obligation": ["ရ", "ပါရစေ"],
            "completion": ["ပြီး", "ခဲ့", "လေ"],
            "continuative": ["နေ", "ဆဲ"]
        }
        
        # ============================================================
        # 2. Tense Markers (ကာလပြစကားလုံးများ)
        # ============================================================
        self.tense_markers = {
            MyanmarTense.PRESENT: {
                "suffixes": ["သည်", "တယ်", "ပါတယ်", "နေသည်", "နေတယ်"],
                "meanings": ["ဖြစ်နေသည်", "လုပ်နေသည်", "ရှိသည်"],
                "examples": ["စားတယ်", "သွားသည်", "နေတယ်"]
            },
            MyanmarTense.PAST: {
                "suffixes": ["ခဲ့သည်", "ခဲ့တယ်", "ခဲ့ဘူး", "လေသည်", "ဖူးသည်"],
                "meanings": ["ဖြစ်ခဲ့သည်", "လုပ်ခဲ့သည်", "ရှိခဲ့သည်"],
                "examples": ["စားခဲ့တယ်", "သွားခဲ့သည်", "ရှိခဲ့ဘူး"]
            },
            MyanmarTense.FUTURE: {
                "suffixes": ["မည်", "လိမ့်မည်", "အုံးမည်", "မယ်", "တော့မယ်"],
                "meanings": ["ဖြစ်လိမ့်မည်", "လုပ်မည်", "ရှိမည်"],
                "examples": ["စားမယ်", "သွားမည်", "လုပ်တော့မယ်"]
            }
        }
        
        # ============================================================
        # 3. Honorifics & Politeness Markers (ယဉ်ကျေးမှုဆိုင်ရာ)
        # ============================================================
        self.honorifics = {
            PolitenessLevel.HIGH: {
                "markers": ["ပါ", "ရှင်", "ခင်ဗျာ", "ကျေးဇူးပြု၍"],
                "pronouns": ["ကျွန်တော်", "ကျွန်မ", "ခင်ဗျား", "ရှင်", "ဒေါ်လေး"],
                "verb_suffixes": ["ပါတယ်", "ပါသည်", "ပါမယ်", "ပါ့မယ်"],
                "descriptions": ["formal", "respectful", "polite"]
            },
            PolitenessLevel.MEDIUM: {
                "markers": [],
                "pronouns": ["ငါ", "မင်း", "သူ", "သူမ"],
                "verb_suffixes": ["တယ်", "သည်", "မယ်"],
                "descriptions": ["standard", "neutral"]
            },
            PolitenessLevel.LOW: {
                "markers": ["ကွ", "ဟေ့", "ရေ", "လေ"],
                "pronouns": ["ငါ", "မင်း", "သူ"],
                "verb_suffixes": ["တယ်ကွ", "တယ်ဟေ့", "မယ်ကွ"],
                "descriptions": ["informal", "casual", "rude"]
            }
        }
        
        # ============================================================
        # 4. Verb (ကြိယာ) - To be (ဖြစ်သည်)
        # ============================================================
        self.verb_to_be = {
            "positive": {
                "present": ["ဖြစ်သည်", "ဖြစ်တယ်", "ပါတယ်"],
                "past": ["ဖြစ်ခဲ့သည်", "ဖြစ်ခဲ့တယ်"],
                "future": ["ဖြစ်မည်", "ဖြစ်လိမ့်မည်", "ဖြစ်မယ်"]
            },
            "negative": {
                "present": ["မဟုတ်ဘူး", "မဖြစ်ဘူး", "မဟုတ်ပါဘူး"],
                "past": ["မဟုတ်ခဲ့ဘူး", "မဖြစ်ခဲ့ဘူး"],
                "future": ["မဖြစ်နိုင်", "မဖြစ်ရ"]
            }
        }
        
        # ============================================================
        # 5. Syntax Order (မြန်မာစာ စကားစုအစီအစဉ်)
        # ============================================================
        # မြန်မာစာမှာ SOV (Subject-Object-Verb) ပုံစံဖြစ်တယ်
        # သို့သော် SVOMPTR က SVO ကို အခြေခံထားတယ်
        # ဒါကြောင့် မြန်မာ Syntax အတိုင်း mapping လုပ်ဖို့လိုတယ်
        
        self.myanmar_syntax_order = ["S", "T", "P", "O", "M", "R", "V"]
        # ပုံမှန်အစီအစဉ်: S + T + P + O + M + R + V
        
        self.slot_particles = {
            "S": "က",
            "O": "ကို",
            "M": "စွာ",
            "P": "မှာ",
            "T": "အချိန်",
            "R": "ကြောင့်"
        }
    
    # ============================================================
    # PARTICLE DETECTION & SLOT MAPPING
    # ============================================================
    
    def detect_particle(self, phrase: str) -> Tuple[str, str, str]:
        """
        Detect particle and map to SVOMPTR slot
        """
        for slot, particles in self.particles.items():
            for particle in particles:
                if phrase.endswith(particle):
                    core_word = phrase[:-len(particle)].strip()
                    return slot, core_word, particle
        
        # No particle found - likely verb or base word
        return "V", phrase, ""
    
    def detect_all_particles(self, tokens: List[str]) -> List[MyanmarSlotMapping]:
        """Detect all particles in token list"""
        mappings = []
        
        for i, token in enumerate(tokens):
            slot, core, particle = self.detect_particle(token)
            if particle:
                mappings.append(MyanmarSlotMapping(
                    original_text=token,
                    slot=slot,
                    particle=particle,
                    core_meaning=core,
                    position=i
                ))
        
        return mappings
    
    # ============================================================
    # TENSE DETECTION
    # ============================================================
    
    def detect_tense(self, verb: str) -> MyanmarVerbInfo:
        """Detect tense from verb suffix"""
        
        # Check negative first (မ...ဘူး pattern)
        is_negative = False
        if verb.startswith("မ") and verb.endswith("ဘူး"):
            is_negative = True
            verb = verb[1:-2]  # Remove မ and ဘူး
        
        # Check each tense
        for tense, info in self.tense_markers.items():
            for suffix in info["suffixes"]:
                if verb.endswith(suffix):
                    root = verb[:-len(suffix)]
                    
                    # Detect politeness
                    politeness = PolitenessLevel.MEDIUM
                    if "ပါ" in suffix:
                        politeness = PolitenessLevel.HIGH
                    
                    return MyanmarVerbInfo(
                        root=root.strip(),
                        suffix=suffix,
                        tense=tense,
                        is_negative=is_negative,
                        politeness=politeness
                    )
        
        # Default: present tense, no suffix found
        return MyanmarVerbInfo(
            root=verb,
            suffix="",
            tense=MyanmarTense.PRESENT,
            is_negative=is_negative
        )
    
    # ============================================================
    # POLITENESS DETECTION
    # ============================================================
    
    def detect_politeness(self, tokens: List[str]) -> PolitenessLevel:
        """Detect politeness level from tokens"""
        
        for level, info in self.honorifics.items():
            for marker in info["markers"]:
                if marker in " ".join(tokens):
                    return level
            for pronoun in info["pronouns"]:
                if pronoun in " ".join(tokens):
                    return level
        
        return PolitenessLevel.MEDIUM
    
    # ============================================================
    # SENTENCE SYNTHESIS (မြန်မာစာ ပြန်ဆောက်ခြင်း)
    # ============================================================
    
    def reconstruct_sentence(self, slots: Dict[str, str], politeness: PolitenessLevel = PolitenessLevel.MEDIUM) -> str:
        """
        Reconstruct Myanmar sentence from SVOMPTR slots
        Follows Myanmar syntax order: S + T + P + O + M + R + V
        """
        sentence_parts = []
        
        for slot in self.myanmar_syntax_order:
            if slot in slots and slots[slot]:
                core = slots[slot]
                particle = self.slot_particles.get(slot, "")
                
                if particle:
                    sentence_parts.append(f"{core}{particle}")
                else:
                    sentence_parts.append(core)
        
        # Add tense suffix to verb (last part)
        if sentence_parts and "V" in slots:
            verb_part = sentence_parts[-1]
            # Add default present tense suffix if missing
            if not any(verb_part.endswith(s) for s in ["သည်", "တယ်", "ပါတယ်"]):
                if politeness == PolitenessLevel.HIGH:
                    verb_part += "ပါတယ်"
                else:
                    verb_part += "တယ်"
                sentence_parts[-1] = verb_part
        
        # Join with spaces
        sentence = " ".join(sentence_parts)
        
        # Add politeness markers if needed
        if politeness == PolitenessLevel.HIGH:
            sentence = sentence.replace("တယ်", "ပါတယ်")
        
        return sentence
    
    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================
    
    def get_tense_name(self, tense: MyanmarTense) -> str:
        """Get Myanmar name for tense"""
        tense_names = {
            MyanmarTense.PRESENT: "ပစ္စုပ္ပန်ကာလ",
            MyanmarTense.PAST: "အတိတ်ကာလ",
            MyanmarTense.FUTURE: "အနာဂတ်ကာလ",
            MyanmarTense.NEGATIVE: "ငြင်းဆိုချက်"
        }
        return tense_names.get(tense, "မသိရှိရ")
    
    def get_politeness_name(self, level: PolitenessLevel) -> str:
        """Get Myanmar name for politeness level"""
        level_names = {
            PolitenessLevel.HIGH: "ယဉ်ကျေးသော (ရိုသေလေးစားမှု)",
            PolitenessLevel.MEDIUM: "ပုံမှန်",
            PolitenessLevel.LOW: "ရင်းနှီးသော (သို့) မယဉ်ကျေးသော"
        }
        return level_names.get(level, "ပုံမှန်")
    
    def analyze_sentence(self, sentence: str) -> Dict:
        """Complete analysis of a Myanmar sentence"""
        tokens = sentence.split()
        
        # Detect particles and map to slots
        particle_mappings = self.detect_all_particles(tokens)
        
        # Detect politeness
        politeness = self.detect_politeness(tokens)
        
        # Find verb and detect tense
        verb_info = None
        for token in tokens:
            # Check if it's potentially a verb (not ending in known particles)
            slot, _, _ = self.detect_particle(token)
            if slot == "V":
                v_info = self.detect_tense(token)
                if v_info.suffix:
                    verb_info = v_info
                    break
        
        # Build slots dictionary
        slots = {}
        for mapping in particle_mappings:
            slots[mapping.slot] = mapping.core_meaning
        
        return {
            "original": sentence,
            "slots": slots,
            "particles": [{"slot": m.slot, "particle": m.particle, "word": m.core_meaning} for m in particle_mappings],
            "tense": self.get_tense_name(verb_info.tense) if verb_info else "မသိရှိရ",
            "politeness": self.get_politeness_name(politeness),
            "is_negative": verb_info.is_negative if verb_info else False
        }
