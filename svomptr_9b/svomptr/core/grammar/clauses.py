# /svomptr_9b/svomptr/core/grammar/clauses.py

"""
Clause Types Handler
Independent, Dependent, Noun, Adjective, Adverb clauses
Based on Myo Min Aung's specification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ClauseType(Enum):
    INDEPENDENT = "independent"      # Main clause
    DEPENDENT = "dependent"          # Subordinate clause
    NOUN_CLAUSE = "noun_clause"      # Acts as noun (S/O)
    ADJECTIVE_CLAUSE = "adj_clause"  # Modifies noun (relative)
    ADVERB_CLAUSE = "adv_clause"     # Modifies verb (time, reason, condition)


@dataclass
class ClauseInfo:
    type: ClauseType
    text: str
    start_pos: int
    end_pos: int
    subordinator: Optional[str] = None
    main_clause: Optional[str] = None


class ClauseHandler:
    """Handles all clause types"""
    
    def __init__(self):
        # Subordinating conjunctions (dependent clause markers)
        self.subordinators = {
            "time": ["when", "while", "as", "before", "after", "until", "since"],
            "reason": ["because", "since", "as", "now that"],
            "condition": ["if", "unless", "provided that", "as long as"],
            "concession": ["although", "though", "even though", "whereas"],
            "purpose": ["so that", "in order that"],
            "result": ["so...that", "such...that"],
            "place": ["where", "wherever"],
            "manner": ["as", "like", "as if", "as though"]
        }
        
        # Relative pronouns (adjective clause markers)
        self.relative_pronouns = ["who", "whom", "whose", "which", "that"]
        
        # Noun clause markers
        self.noun_clause_markers = ["what", "whatever", "who", "whoever", "whom", 
                                     "which", "whichever", "that", "whether", "if"]
    
    def detect_clauses(self, tokens: List[str]) -> List[ClauseInfo]:
        """
        Detect all clauses in sentence
        """
        clauses = []
        sentence = " ".join(tokens)
        
        # 1. Detect dependent clauses (subordinator + clause)
        for category, words in self.subordinators.items():
            for word in words:
                if f" {word} " in f" {sentence} ":
                    # Find clause after subordinator
                    parts = sentence.split(word, 1)
                    if len(parts) == 2:
                        clause_text = parts[1].strip()
                        # Find where clause ends (look for period or main clause)
                        end_markers = [".", "!", "?", " and ", " but ", " or ", " so "]
                        for marker in end_markers:
                            if marker in clause_text:
                                clause_text = clause_text.split(marker)[0]
                                break
                        
                        clauses.append(ClauseInfo(
                            type=ClauseType.DEPENDENT,
                            text=clause_text,
                            start_pos=sentence.find(word),
                            end_pos=sentence.find(word) + len(word) + len(clause_text),
                            subordinator=word
                        ))
        
        # 2. Detect relative clauses (noun + relative pronoun + clause)
        for i, token in enumerate(tokens):
            if token.lower() in self.relative_pronouns:
                # Find the noun it modifies
                noun = tokens[i-1] if i > 0 else None
                # Extract relative clause
                clause_text = " ".join(tokens[i:])
                clauses.append(ClauseInfo(
                    type=ClauseType.ADJECTIVE_CLAUSE,
                    text=clause_text,
                    start_pos=i,
                    end_pos=len(tokens),
                    subordinator=token,
                    main_clause=noun
                ))
        
        # 3. Detect noun clauses (wh-word + clause as S/O)
        for i, token in enumerate(tokens):
            if token.lower() in self.noun_clause_markers:
                # Check if it acts as subject or object
                # Find main verb after clause
                clause_text = " ".join(tokens[i:])
                # Find where clause ends (before main verb)
                main_verbs = ["is", "are", "was", "were", "seems", "appears"]
                for v in main_verbs:
                    if f" {v} " in f" {clause_text} ":
                        clause_text = clause_text.split(v)[0].strip()
                        break
                
                clauses.append(ClauseInfo(
                    type=ClauseType.NOUN_CLAUSE,
                    text=clause_text,
                    start_pos=i,
                    end_pos=i + len(clause_text.split()),
                    subordinator=token
                ))
        
        return clauses
    
    def classify_clause(self, clause: str) -> ClauseType:
        """Classify clause type based on structure"""
        clause_lower = clause.lower()
        
        # Check for subordinators
        for category, words in self.subordinators.items():
            for word in words:
                if clause_lower.startswith(word):
                    return ClauseType.DEPENDENT
        
        # Check for relative pronouns
        for rp in self.relative_pronouns:
            if rp in clause_lower.split():
                return ClauseType.ADJECTIVE_CLAUSE
        
        # Check for noun clause markers
        for nm in self.noun_clause_markers:
            if clause_lower.startswith(nm):
                return ClauseType.NOUN_CLAUSE
        
        return ClauseType.INDEPENDENT
    
    def get_clause_rules(self) -> Dict:
        """Get clause usage rules"""
        return {
            "independent": {
                "meaning": "အဓိကဝါကျ - တစ်ခုတည်းရပ်တည်နိုင်သော",
                "example": "I went home"
            },
            "dependent": {
                "meaning": "လက်အောက်ခံဝါကျ - အဓိကဝါကျမပါဘဲ မရပ်တည်နိုင်",
                "example": "because I was tired",
                "subordinators": self.subordinators
            },
            "noun_clause": {
                "meaning": "နာမ်စားသဖွယ် ဆောင်ရွက်သော အပိုဒ်",
                "example": "What he said is true"
            },
            "adjective_clause": {
                "meaning": "နာမ်ကို ပြုပြင်ပေးသော အပိုဒ်",
                "example": "The book that I read",
                "relative_pronouns": self.relative_pronouns
            },
            "adverb_clause": {
                "meaning": "ကြိယာကို ပြုပြင်ပေးသော အပိုဒ်",
                "example": "When I arrived, he left"
            }
        }
