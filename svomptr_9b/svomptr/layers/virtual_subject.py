"""
Virtual Subject Handler for "There is" and "It is"
Based on Myo Min Aung's specification
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VirtualSubjectInfo:
    """Complete virtual subject information"""
    is_virtual: bool
    type: str  # "existential", "dummy_weather", "dummy_general", "none"
    word: str  # "it", "there", or ""
    verb: Optional[str] = None
    introduced_noun: Optional[str] = None  # For "There is X"
    refers_to: Optional[str] = None  # For "It" resolution
    position: int = -1


class VirtualSubjectDetector:
    """
    Detect "There is/are" and "It is" patterns
    CRITICAL for proper SVOMPTR parsing
    """
    
    def __init__(self):
        # Weather verbs (stand alone with "it") - NO REAL SUBJECT
        self.weather_verbs = {
            "en": ["rain", "rains", "raining", "snow", "snows", "snowing", 
                   "storm", "storms", "storming", "hail", "thunder", "lightning"],
            "my": ["ရွာ", "ရွာတယ်", "ရွာနေ", "နှင်း", "နှင်းကျ", 
                   "မုန်တိုင်း", "လေ", "မိုး", "ပူ", "အေး"]
        }
        
        # Existence verbs (for "there") - INTRODUCES NEW SUBJECT
        self.existence_verbs = {
            "en": ["is", "are", "was", "were", "exists", "remains", "stands", "lies"],
            "my": ["ရှိ", "ရှိသည်", "ရှိနေ", "တည်ရှိ", "ပေါ်လာ", "ဖြစ်ပေါ်"]
        }
        
        # Identity verbs (for "it is") - REFERS TO PREVIOUS SUBJECT
        self.identity_verbs = {
            "en": ["is", "was", "seems", "appears", "looks", "feels", "sounds", "tastes"],
            "my": ["ဖြစ်", "ဖြစ်သည်", "ဖြစ်နေ", "မည်", "ဟု", "ကဲ့သို့"]
        }
        
        # Context for pronoun resolution (MUST track across sentences)
        self.last_introduced_noun = None
        self.last_subject = None
        self.conversation_history = []
    
    def detect(self, tokens: List[str]) -> Tuple[bool, VirtualSubjectInfo]:
        """
        Detect virtual subject pattern in tokens
        
        CRITICAL RULES:
        1. "There is" + existence verb = EXISTENTIAL (introduces new subject)
        2. "It is" + weather verb = DUMMY WEATHER (no subject)
        3. "It is" + identity verb = DUMMY GENERAL (refers to previous)
        
        Returns (is_virtual, VirtualSubjectInfo)
        """
        if not tokens:
            return False, VirtualSubjectInfo(is_virtual=False, type="none", word="")
        
        first = tokens[0].lower()
        
        # ============================================================
        # RULE 1: "There is/are/was/were" → EXISTENTIAL
        # Example: "There is a cat" → introduces "cat"
        # ============================================================
        if first == "there" and len(tokens) > 1:
            verb = tokens[1].lower()
            
            # Check if it's an existence verb
            if verb in self.existence_verbs["en"] or verb in self.existence_verbs["my"]:
                # Extract the introduced noun (what exists)
                introduced = None
                if len(tokens) > 2:
                    # Skip determiners (a, an, the)
                    idx = 2
                    while idx < len(tokens) and tokens[idx] in ["a", "an", "the", "some", "any"]:
                        idx += 1
                    if idx < len(tokens):
                        introduced = tokens[idx]
                
                # CRITICAL: Store for future "it" resolution
                if introduced:
                    self.last_introduced_noun = introduced
                    self.last_subject = introduced
                
                return True, VirtualSubjectInfo(
                    is_virtual=True,
                    type="existential",
                    word="there",
                    verb=verb,
                    introduced_noun=introduced,
                    position=0
                )
        
        # ============================================================
        # RULE 2: "It is" + weather verb → DUMMY WEATHER
        # Example: "It is raining" → no real subject
        # ============================================================
        if first == "it" and len(tokens) > 1:
            verb = tokens[1].lower()
            
            # Check if it's a weather verb (NO real subject)
            if verb in self.weather_verbs["en"] or verb in self.weather_verbs["my"]:
                return True, VirtualSubjectInfo(
                    is_virtual=True,
                    type="dummy_weather",
                    word="it",
                    verb=verb,
                    position=0
                )
            
            # Check if it's an identity verb (REFERS to previous)
            if verb in self.identity_verbs["en"] or verb in self.identity_verbs["my"]:
                # CRITICAL: Resolve what "it" refers to
                refers_to = self.last_introduced_noun or self.last_subject
                
                return True, VirtualSubjectInfo(
                    is_virtual=True,
                    type="dummy_general",
                    word="it",
                    verb=verb,
                    refers_to=refers_to,
                    position=0
                )
        
        return False, VirtualSubjectInfo(is_virtual=False, type="none", word="")
    
    def resolve_it(self, virtual_info: VirtualSubjectInfo) -> Optional[str]:
        """
        Resolve what "it" refers to
        Called when "it" is used as a general reference
        """
        if virtual_info.type != "dummy_general":
            return None
        
        # Priority 1: Direct reference from detection
        if virtual_info.refers_to:
            return virtual_info.refers_to
        
        # Priority 2: Last introduced noun
        if self.last_introduced_noun:
            return self.last_introduced_noun
        
        # Priority 3: Last subject
        return self.last_subject
    
    def update_context(self, subject: str, noun_phrase: str = None):
        """Update context for pronoun resolution across sentences"""
        if subject:
            self.last_subject = subject
        if noun_phrase:
            self.last_introduced_noun = noun_phrase
    
    def clear_context(self):
        """Clear context for new conversation"""
        self.last_introduced_noun = None
        self.last_subject = None
        self.conversation_history = []


class VirtualSubjectEmbedding(nn.Module):
    """
    Special learnable embeddings for virtual subjects
    Different types get different embeddings
    """
    
    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable embeddings for each virtual type
        # These are PARAMETERS that will be trained
        self.existential_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.dummy_weather_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.dummy_general_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Projection layers
        self.existential_proj = nn.Linear(hidden_dim, hidden_dim)
        self.weather_proj = nn.Linear(hidden_dim, hidden_dim)
        self.general_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, virtual_info: VirtualSubjectInfo, position: int = 0) -> torch.Tensor:
        """
        Get embedding for virtual subject token
        
        Args:
            virtual_info: Virtual subject information
            position: Position in sequence
            
        Returns:
            embedding: (1, 1, hidden_dim)
        """
        if virtual_info.type == "existential":
            emb = self.existential_embed
            emb = self.existential_proj(emb)
        elif virtual_info.type == "dummy_weather":
            emb = self.dummy_weather_embed
            emb = self.weather_proj(emb)
        elif virtual_info.type == "dummy_general":
            emb = self.dummy_general_embed
            emb = self.general_proj(emb)
        else:
            emb = torch.zeros(1, 1, self.hidden_dim)
        
        # Add position information
        pos = position
        d = self.hidden_dim
        pe = torch.zeros(d, device=emb.device)
        div_term = torch.exp(torch.arange(0, d, 2, device=emb.device) * -(math.log(10000.0) / d))
        pe[0::2] = torch.sin(pos * div_term)
        pe[1::2] = torch.cos(pos * div_term[:d//2])
        emb = emb + pe.view(1, 1, -1)
        
        return emb
