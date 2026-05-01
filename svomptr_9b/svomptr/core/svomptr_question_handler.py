"""
SVOMPTR Question Handler
Handles: Questions, Wh-clauses, Relative clauses, Modal verbs
Based on Myo Min Aung's complete specification
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class QuestionType(Enum):
    YES_NO = "yes_no"           # "Do you like it?"
    WH_QUESTION = "wh_question"  # "What is this?"
    TAG_QUESTION = "tag"         # "You like it, don't you?"
    PERMISSION = "permission"    # "May I come in?"
    RHETORICAL = "rhetorical"    # "Who knows?"


class ClauseType(Enum):
    MAIN = "main"
    WH_CLAUSE = "wh_clause"      # "What he said" as subject
    RELATIVE_CLAUSE = "relative"  # "The book that I read"
    GERUND_PHRASE = "gerund"      # "Running is fun"
    INFINITIVE_PHRASE = "infinitive"  # "To be or not to be"


@dataclass
class QuestionInfo:
    is_question: bool = False
    question_type: QuestionType = QuestionType.YES_NO
    question_word: Optional[str] = None  # what, why, who, when, where, how
    auxiliary_verb: Optional[str] = None  # do, does, did, has, have, had
    modal_verb: Optional[str] = None      # will, would, shall, should, may, might, must
    is_permission: bool = False
    has_please: bool = False


@dataclass
class ClauseInfo:
    clause_type: ClauseType = ClauseType.MAIN
    is_embedded: bool = False  # clause inside another clause
    is_subject_clause: bool = False  # clause acting as S
    is_object_clause: bool = False   # clause acting as O
    relative_pronoun: Optional[str] = None  # who, which, that, where
    wh_word: Optional[str] = None  # what, why, who, when, where, how


class SVOMPTRQuestionHandler:
    """
    Handles ALL question patterns and clauses
    """
    
    def __init__(self):
        # Question words
        self.question_words = {
            "what", "why", "who", "whom", "whose", "which", "when", "where", "how"
        }
        
        # Auxiliary verbs (for questions)
        self.auxiliary_verbs = {
            "do", "does", "did", "have", "has", "had", "is", "am", "are", "was", "were"
        }
        
        # Modal verbs
        self.modal_verbs = {
            "will", "would", "shall", "should", "may", "might", "must", "can", "could"
        }
        
        # Relative pronouns
        self.relative_pronouns = {
            "who", "whom", "whose", "which", "that", "where", "when"
        }
    
    def detect_question(self, tokens: List[str]) -> QuestionInfo:
        """Detect if sentence is a question and what type"""
        
        if not tokens:
            return QuestionInfo()
        
        info = QuestionInfo()
        first = tokens[0].lower().strip('!?,.')
        second = tokens[1].lower().strip('!?,.') if len(tokens) > 1 else ""
        
        # Check for "please"
        if any("please" in t.lower() for t in tokens):
            info.has_please = True
        
        # Permission request logic
        if first in ["may", "can", "could"] and second in ["i", "we", "you"]:
            info.is_question = True
            info.question_type = QuestionType.PERMISSION
            info.modal_verb = first
            info.is_permission = True
            return info
        
        # Wh-question logic
        if first in self.question_words:
            info.is_question = True
            info.question_type = QuestionType.WH_QUESTION
            info.question_word = first
            
            if len(tokens) > 1:
                if second in self.auxiliary_verbs:
                    info.auxiliary_verb = second
                elif second in self.modal_verbs:
                    info.modal_verb = second
            return info
        
        # Yes/No question logic
        if first in self.auxiliary_verbs or first in self.modal_verbs:
            if len(tokens) > 1 and second not in self.question_words:
                info.is_question = True
                info.question_type = QuestionType.YES_NO
                if first in self.auxiliary_verbs:
                    info.auxiliary_verb = first
                else:
                    info.modal_verb = first
                return info
        
        # Final fallback - punctuation
        if tokens[-1].endswith('?'):
            info.is_question = True
            
        return info


class SVOMPTRClauseHandler:
    """
    Handles clauses that act as S or O
    """
    
    def __init__(self):
        self.wh_words = {"what", "why", "who", "whom", "whose", "which", "when", "where", "how"}
        self.relative_pronouns = {"who", "whom", "whose", "which", "that", "where", "when"}
        
    def detect_noun_clause(self, tokens: List[str]) -> Optional[ClauseInfo]:
        """Detect if a clause acts as a noun (S or O)"""
        if not tokens:
            return None
        
        info = ClauseInfo()
        first = tokens[0].lower().strip('!?,.')
        
        # Wh-clause at start (likely Subject)
        if first in self.wh_words and len(tokens) > 2:
            # Simple heuristic: if a verb follows later, the whole previous part is S
            # e.g., "What he said is true"
            info.is_embedded = True
            info.clause_type = ClauseType.WH_CLAUSE
            info.wh_word = first
            info.is_subject_clause = True
            return info
            
        # Relative clause detector
        for i, token in enumerate(tokens):
            if token.lower() in self.relative_pronouns and i > 0:
                info.is_embedded = True
                info.clause_type = ClauseType.RELATIVE_CLAUSE
                info.relative_pronoun = token.lower()
                return info
        
        return None

class SVOMPTRCompleteParser:
    """
    Integrated parser for Questions, Clauses, and Modals
    """
    def __init__(self):
        self.question_handler = SVOMPTRQuestionHandler()
        self.clause_handler = SVOMPTRClauseHandler()
        
    def parse(self, sentence: str) -> Dict:
        tokens = sentence.strip().split()
        if not tokens: return {}
        
        q_info = self.question_handler.detect_question(tokens)
        c_info = self.clause_handler.detect_noun_clause(tokens)
        
        result = {
            "is_question": q_info.is_question,
            "question_type": q_info.question_type.value if q_info.is_question else None,
            "is_permission": q_info.is_permission,
            "has_clause": c_info.is_embedded if c_info else False,
            "clause_type": c_info.clause_type.value if (c_info and c_info.is_embedded) else None,
            "S": None, "V": None, "O": None
        }
        
        # Simple extraction logic for demo
        if not q_info.is_question:
            if c_info and c_info.is_subject_clause:
                # Heuristic: split by main verbs
                main_verbs = ["is", "are", "was", "were", "seems"]
                split_idx = -1
                for i, t in enumerate(tokens):
                    if t.lower() in main_verbs and i > 0:
                        split_idx = i
                        break
                if split_idx != -1:
                    result["S"] = " ".join(tokens[:split_idx])
                    result["V"] = tokens[split_idx]
                    result["O"] = " ".join(tokens[split_idx+1:])
            else:
                result["S"] = tokens[0]
                result["V"] = tokens[1] if len(tokens) > 1 else None
                result["O"] = " ".join(tokens[2:]) if len(tokens) > 2 else None
        
        return result
