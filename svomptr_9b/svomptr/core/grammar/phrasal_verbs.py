# /svomptr_9b/svomptr/core/grammar/phrasal_verbs.py

"""
Phrasal Verbs Handler
Handles all phrasal verb patterns (verb + particle)
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PhrasalVerbType(Enum):
    INTRANSITIVE = "intransitive"          # give up (no object)
    TRANSITIVE_SEPARABLE = "transitive_separable"    # turn on the light / turn the light on
    TRANSITIVE_INSEPARABLE = "transitive_inseparable" # look after the baby


@dataclass
class PhrasalVerbInfo:
    is_phrasal: bool
    verb: str
    particle: str
    type: PhrasalVerbType
    meaning: str
    object: Optional[str] = None
    object_position: str = ""  # "between" or "after"


class PhrasalVerbHandler:
    """
    Handles ALL phrasal verb patterns
    """
    
    def __init__(self):
        # ============================================================
        # Intransitive phrasal verbs (no object)
        # ============================================================
        self.intransitive = {
            "give up": {"meaning": "လက်လျှော့သည်", "example": "He gave up"},
            "show up": {"meaning": "ပေါ်လာသည်", "example": "She showed up late"},
            "wake up": {"meaning": "နိုးသည်", "example": "I wake up early"},
            "stand up": {"meaning": "မတ်တပ်ရပ်သည်", "example": "Please stand up"},
            "sit down": {"meaning": "ထိုင်သည်", "example": "Sit down please"},
            "get up": {"meaning": "ထသည်", "example": "I get up at 7"},
            "go out": {"meaning": "ထွက်သွားသည်", "example": "Let's go out"},
            "come back": {"meaning": "ပြန်လာသည်", "example": "He came back"},
            "hurry up": {"meaning": "မြန်သည်", "example": "Hurry up!"},
            "slow down": {"meaning": "နှေးသည်", "example": "Slow down please"},
            "settle down": {"meaning": "အခြေချသည်", "example": "They settled down"},
            "grow up": {"meaning": "ကြီးပြင်းသည်", "example": "She grew up in Yangon"}
        }
        
        # ============================================================
        # Transitive separable phrasal verbs (can separate)
        # ============================================================
        self.transitive_separable = {
            "turn on": {"meaning": "ဖွင့်သည်", "example": "Turn on the light / Turn the light on"},
            "turn off": {"meaning": "ပိတ်သည်", "example": "Turn off the TV / Turn the TV off"},
            "put on": {"meaning": "ဝတ်ဆင်သည်", "example": "Put on your coat / Put your coat on"},
            "take off": {"meaning": "ချွတ်သည်", "example": "Take off your shoes / Take your shoes off"},
            "pick up": {"meaning": "ကောက်သည်", "example": "Pick up the book / Pick the book up"},
            "drop off": {"meaning": "ချပေးသည်", "example": "Drop off the package / Drop the package off"},
            "throw away": {"meaning": "ပစ်သည်", "example": "Throw away the trash / Throw the trash away"},
            "clean up": {"meaning": "သန့်ရှင်းသည်", "example": "Clean up the room / Clean the room up"},
            "fill out": {"meaning": "ဖြည့်သည်", "example": "Fill out the form / Fill the form out"},
            "look up": {"meaning": "ရှာသည်", "example": "Look up the word / Look the word up"},
            "call off": {"meaning": "ဖျက်သည်", "example": "Call off the meeting / Call the meeting off"},
            "put away": {"meaning": "သိမ်းသည်", "example": "Put away your toys / Put your toys away"}
        }
        
        # ============================================================
        # Transitive inseparable phrasal verbs (cannot separate)
        # ============================================================
        self.transitive_inseparable = {
            "look after": {"meaning": "စောင့်ရှောက်သည်", "example": "Look after the baby"},
            "look for": {"meaning": "ရှာဖွေသည်", "example": "Look for your keys"},
            "run into": {"meaning": "တွေ့သည်", "example": "I ran into an old friend"},
            "get over": {"meaning": "ကျော်လွှားသည်", "example": "Get over your fear"},
            "go over": {"meaning": "ပြန်လေ့လာသည်", "example": "Go over the lesson"},
            "come across": {"meaning": "တွေ့ရှိသည်", "example": "I came across an old photo"},
            "deal with": {"meaning": "ကိုင်တွယ်သည်", "example": "Deal with the problem"},
            "get along with": {"meaning": "သင့်မြတ်သည်", "example": "Get along with your colleagues"},
            "put up with": {"meaning": "သည်းခံသည်", "example": "Put up with the noise"},
            "stand for": {"meaning": "ကိုယ်စားပြုသည်", "example": "What does VIP stand for?"},
            "care for": {"meaning": "ကြိုက်သည်", "example": "I don't care for coffee"},
            "count on": {"meaning": "ယုံကြည်အားထားသည်", "example": "You can count on me"}
        }
        
        # Particles/Prepositions that form phrasal verbs
        self.particles = ["up", "down", "on", "off", "in", "out", "over", "under", 
                          "away", "back", "forward", "through", "across", "along", 
                          "against", "into", "after", "for", "with", "without"]
    
    def detect_phrasal_verb(self, tokens: List[str]) -> PhrasalVerbInfo:
        """
        Detect phrasal verb in sentence
        """
        if len(tokens) < 2:
            return PhrasalVerbInfo(False, "", "", None, "")
        
        # Check for two-word phrasal verbs
        for i in range(len(tokens) - 1):
            two_word = f"{tokens[i].lower()} {tokens[i+1].lower()}"
            
            # Check intransitive
            if two_word in self.intransitive:
                return PhrasalVerbInfo(
                    is_phrasal=True,
                    verb=tokens[i],
                    particle=tokens[i+1],
                    type=PhrasalVerbType.INTRANSITIVE,
                    meaning=self.intransitive[two_word]["meaning"]
                )
            
            # Check transitive separable (verb + particle)
            if two_word in self.transitive_separable:
                # Check if object is between or after
                obj = None
                obj_position = ""
                
                # Pattern: verb + particle + object
                if i + 2 < len(tokens):
                    obj = tokens[i+2]
                    obj_position = "after"
                else:
                    obj_position = "none"
                
                return PhrasalVerbInfo(
                    is_phrasal=True,
                    verb=tokens[i],
                    particle=tokens[i+1],
                    type=PhrasalVerbType.TRANSITIVE_SEPARABLE,
                    meaning=self.transitive_separable[two_word]["meaning"],
                    object=obj,
                    object_position=obj_position
                )
            
            # Check transitive inseparable
            if two_word in self.transitive_inseparable:
                obj = tokens[i+2] if i + 2 < len(tokens) else None
                return PhrasalVerbInfo(
                    is_phrasal=True,
                    verb=tokens[i],
                    particle=tokens[i+1],
                    type=PhrasalVerbType.TRANSITIVE_INSEPARABLE,
                    meaning=self.transitive_inseparable[two_word]["meaning"],
                    object=obj,
                    object_position="after"
                )
        
        # Check for separated separable phrasal verbs (verb + object + particle)
        for i, token in enumerate(tokens):
            if i + 2 < len(tokens):
                verb = token.lower()
                particle = tokens[i+2].lower()
                
                for pv, info in self.transitive_separable.items():
                    parts = pv.split()
                    if len(parts) == 2 and verb == parts[0] and particle == parts[1]:
                        obj = tokens[i+1]
                        return PhrasalVerbInfo(
                            is_phrasal=True,
                            verb=tokens[i],
                            particle=tokens[i+2],
                            type=PhrasalVerbType.TRANSITIVE_SEPARABLE,
                            meaning=info["meaning"],
                            object=obj,
                            object_position="between"
                        )
        
        return PhrasalVerbInfo(False, "", "", None, "")
    
    def get_phrasal_verb_rules(self) -> Dict:
        """Get phrasal verb usage rules"""
        return {
            "intransitive": {
                "ဖွဲ့စည်းပုံ": "Verb + Particle",
                "example": "He gave up. (သူလက်လျှော့လိုက်သည်)",
                "note": "Object မလိုအပ်"
            },
            "transitive_separable": {
                "ဖွဲ့စည်းပုံ": "Verb + Particle + Object OR Verb + Object + Particle",
                "example": "Turn on the light / Turn the light on",
                "note": "Particle ကို ရွှေ့ပြောင်းနိုင်သည်"
            },
            "transitive_inseparable": {
                "ဖွဲ့စည်းပုံ": "Verb + Particle + Object (ရွှေ့မရ)",
                "example": "Look after the baby",
                "note": "Particle ကို မရွှေ့ရ"
            }
        }
    
    def get_meaning(self, phrasal_verb: str) -> str:
        """Get Myanmar meaning of phrasal verb"""
        if phrasal_verb in self.intransitive:
            return self.intransitive[phrasal_verb]["meaning"]
        if phrasal_verb in self.transitive_separable:
            return self.transitive_separable[phrasal_verb]["meaning"]
        if phrasal_verb in self.transitive_inseparable:
            return self.transitive_inseparable[phrasal_verb]["meaning"]
        return ""
    
    def get_all_phrasal_verbs(self) -> Dict:
        """Get all phrasal verbs categorized"""
        return {
            "intransitive": list(self.intransitive.keys()),
            "transitive_separable": list(self.transitive_separable.keys()),
            "transitive_inseparable": list(self.transitive_inseparable.keys())
        }
