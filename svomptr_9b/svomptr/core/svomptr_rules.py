"""
SVOMPTR Complete Rules - Based on Myo Min Aung's Specification
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class SentenceType(Enum):
    IMPERATIVE = "imperative"      # "Sit down" - no subject
    DECLARATIVE = "declarative"    # Normal sentence
    INTERJECTION = "interjection"  # "Wow!" - emotion only
    EXISTENTIAL = "existential"    # "There is"
    WEATHER = "weather"            # "It is raining"


@dataclass
class SVOMPTRFrame:
    """Complete SVOMPTR frame with all rules applied"""
    sentence_type: SentenceType = SentenceType.DECLARATIVE
    
    # Core slots
    S: Optional[str] = None      # Subject (can be implicit)
    V: Optional[str] = None      # Verb
    O: Optional[str] = None      # Object (optional for intransitive)
    M: Optional[str] = None      # Manner/Emotion (including interjections)
    P: Optional[str] = None      # Place
    T: Optional[str] = None      # Time
    R: Optional[str] = None      # Reason (to-infinitive, because clause)
    
    # Metadata
    is_imperative: bool = False
    has_implied_subject: bool = False
    implied_subject: str = "you"
    is_interjection: bool = False
    interjection_type: Optional[str] = None
    is_intransitive: bool = False  # Verb that doesn't need object
    has_to_infinitive: bool = False
    has_gerund: bool = False
    ing_is_manner: bool = False
    
    # For storing parsed components
    raw_text: str = ""
    tokens: List[str] = field(default_factory=list)


class SVOMPTRRuleEngine:
    """
    Implements all SVOMPTR rules from Myo Min Aung's specification
    """
    
    def __init__(self):
        # Intransitive verbs (ကမလို ကြိယာများ - O မလို)
        self.intransitive_verbs = {
            "en": ["sleep", "sleeps", "slept", "run", "runs", "ran", "arrive", "arrives", 
                   "arrived", "sit", "sits", "sat", "stand", "stands", "stood", "lie", 
                   "lies", "lay", "die", "dies", "died", "laugh", "cry", "smile"],
            "my": ["အိပ်", "ပြေး", "ရောက်", "ထိုင်", "ရပ်", "အော်", "ရယ်", "ငို"]
        }
        
        # Imperative verbs (အာဏာပြု - S implied as "you")
        self.imperative_verbs = {
            "en": ["go", "come", "sit", "stand", "run", "stop", "start", "eat", "drink",
                   "read", "write", "listen", "look", "watch", "help", "beat", "hit"],
            "my": ["သွား", "လာ", "ထိုင်", "ရပ်", "စား", "သောက်", "ဖတ်", "ရေး", "ကူ"]
        }
        
        # Interjections (အာမေဋိတ်များ - go to M slot)
        self.interjections = {
            "positive": ["wow", "great", "awesome", "excellent", "အား", "အေး", "ဟုတ်"],
            "negative": ["oh", "alas", "sorry", "oops", "အို", "အားး", "အမလေး"],
            "emotion": ["happy", "sad", "excited", "tired", "ပျော်", "ဝမ်းနည်း", "စိတ်လှုပ်ရှား"]
        }
        
        # To-infinitive markers (go to R slot)
        self.to_infinitive_markers = ["to", "in order to", "so as to", "ဖို့", "ရန်"]
        
        # -ing forms that can be Manner
        self.ing_manner_verbs = ["run", "running", "cry", "crying", "laugh", "laughing",
                                  "walk", "walking", "jump", "jumping", "smile", "smiling",
                                  "ပြေး", "ပြေးနေ", "ငို", "ငိုနေ", "ရယ်", "ရယ်နေ"]
        
        # Gerund markers (can be S or O)
        self.gerund_markers = ["ing", "နေ", "ခြင်း", "မှု"]
    
    def detect_sentence_type(self, tokens: List[str]) -> Tuple[SentenceType, Dict]:
        """Detect sentence type based on rules"""
        
        if not tokens:
            return SentenceType.DECLARATIVE, {}
        
        first = tokens[0].lower()
        
        # Check interjection
        for cat, words in self.interjections.items():
            if first in words or first.strip('!?,.') in words:
                return SentenceType.INTERJECTION, {"type": cat, "word": first}
        
        # Check imperative (no subject, verb first)
        if first in self.imperative_verbs["en"] or first in self.imperative_verbs["my"]:
            return SentenceType.IMPERATIVE, {"verb": first}
        
        # Check existential
        if first == "there" and len(tokens) > 1 and tokens[1] in ["is", "are", "was", "were"]:
            return SentenceType.EXISTENTIAL, {}
        
        # Check weather (it is + weather)
        if first == "it" and len(tokens) > 1 and tokens[1] in ["is", "was", "were"]:
            weather_check = tokens[2] if len(tokens) > 2 else ""
            if any(w in weather_check for w in ["rain", "snow", "hot", "cold", "storm", "wind"]):
                return SentenceType.WEATHER, {}
        
        return SentenceType.DECLARATIVE, {}
    
    def is_intransitive(self, verb: str) -> bool:
        """Check if verb doesn't need object"""
        v = verb.lower().strip('!?,.')
        return v in self.intransitive_verbs["en"] or v in self.intransitive_verbs["my"]
    
    def has_to_infinitive(self, tokens: List[str], start_pos: int = 0) -> Optional[Tuple[int, str]]:
        """Find to-infinitive clause"""
        for i in range(start_pos, len(tokens)):
            if tokens[i].lower() in self.to_infinitive_markers:
                # Extract the infinitive clause
                infinitive = " ".join(tokens[i:])
                return (i, infinitive)
        return None
    
    def is_gerund(self, word: str) -> bool:
        """Check if word is gerund form"""
        word_lower = word.lower()
        return word_lower.endswith("ing") or any(m in word_lower for m in self.gerund_markers)
    
    def is_ing_manner(self, phrase: str) -> bool:
        """Check if -ing phrase is manner (how the action is done)"""
        phrase_lower = phrase.lower()
        for verb in self.ing_manner_verbs:
            if verb in phrase_lower:
                return True
        return False
    
    def parse(self, sentence: str) -> SVOMPTRFrame:
        """Complete parse with all rules applied"""
        
        tokens = sentence.strip().split()
        frame = SVOMPTRFrame(raw_text=sentence, tokens=tokens)
        
        if not tokens:
            return frame

        # Step 1: Detect sentence type
        sent_type, extra = self.detect_sentence_type(tokens)
        frame.sentence_type = sent_type
        
        # Step 2: Handle INTERJECTION
        if sent_type == SentenceType.INTERJECTION:
            frame.is_interjection = True
            frame.M = extra.get("word", tokens[0])
            return frame
        
        # Step 3: Handle IMPERATIVE (S is implied "you")
        if sent_type == SentenceType.IMPERATIVE:
            frame.is_imperative = True
            frame.has_implied_subject = True
            frame.S = "you"
            frame.V = extra.get("verb", tokens[0])
            
            # Rest are object or manner
            remaining = tokens[1:]
            if remaining:
                # Check for object or manner
                if len(remaining) == 1:
                    if self.is_intransitive(frame.V):
                        frame.M = remaining[0]
                    else:
                        frame.O = remaining[0]
                else:
                    frame.O = remaining[0]
                    frame.M = " ".join(remaining[1:])
            return frame
        
        # Step 4: Handle EXISTENTIAL
        if sent_type == SentenceType.EXISTENTIAL:
            frame.S = "there"
            frame.V = tokens[1] if len(tokens) > 1 else "is"
            frame.O = " ".join(tokens[2:]) if len(tokens) > 2 else None
            return frame
        
        # Step 5: Handle WEATHER
        if sent_type == SentenceType.WEATHER:
            frame.S = "it"
            frame.V = tokens[1]
            frame.M = " ".join(tokens[2:]) if len(tokens) > 2 else ""
            return frame
        
        # Step 6: Normal DECLARATIVE parsing
        if len(tokens) < 2:
            frame.S = tokens[0]
            return frame
        
        frame.S = tokens[0]
        frame.V = tokens[1]
        
        # Check if verb is intransitive (O not needed)
        if self.is_intransitive(frame.V):
            frame.is_intransitive = True
            remaining = tokens[2:]
            if remaining:
                # Check for manner or reason
                infinitive_info = self.has_to_infinitive(tokens, 2)
                if infinitive_info:
                    pos, infinitive = infinitive_info
                    frame.R = infinitive
                    frame.has_to_infinitive = True
                    if pos > 2:
                        frame.M = " ".join(tokens[2:pos])
                else:
                    frame.M = " ".join(remaining)
        else:
            # Transitive - needs object
            if len(tokens) > 2:
                # Check for to-infinitive
                infinitive_info = self.has_to_infinitive(tokens, 2)
                if infinitive_info:
                    pos, infinitive = infinitive_info
                    frame.R = infinitive
                    frame.has_to_infinitive = True
                    # Object is before infinitive if any
                    if pos > 2:
                        frame.O = " ".join(tokens[2:pos])
                else:
                    frame.O = tokens[2]
                    if len(tokens) > 3:
                        frame.M = " ".join(tokens[3:])
            else:
                # Might be incomplete or S V only
                pass
        
        # Step 7: Handle -ing as manner
        if frame.M and self.is_ing_manner(frame.M):
            frame.ing_is_manner = True
        
        return frame
