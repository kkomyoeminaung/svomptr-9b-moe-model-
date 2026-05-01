# /svomptr_9b/svomptr/core/svomptr_complete.py

"""
Complete SVOMPTR Parser with All Grammar Rules
Integrates every grammar component
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .grammar.voice import VoiceHandler, Voice
from .grammar.tense import TenseHandler, Tense
from .grammar.conditional import ConditionalHandler, ConditionalType
from .grammar.reported_speech import ReportedSpeechHandler
from .grammar.conjunctions import ConjunctionHandler
from .grammar.negation import NegationHandler
from .grammar.causative import CausativeHandler
from .grammar.ellipsis import EllipsisHandler
from .grammar.discourse import DiscourseHandler
from .grammar.emphasis import EmphasisHandler
from .grammar.prepositions import PrepositionHandler
from .grammar.determiners import DeterminerHandler
from .grammar.numerals import NumeralsHandler
from .grammar.phrasal_verbs import PhrasalVerbHandler
from .grammar.subjunctive import SubjunctiveHandler
from .grammar.reflexive import ReflexiveHandler
from .grammar.adverbs import AdverbsHandler
from .grammar.clauses import ClauseHandler
from .grammar.appositives import AppositiveHandler
from .grammar.tag_questions import TagQuestionHandler
from .grammar.absolute_phrases import AbsolutePhraseHandler
from .grammar.punctuation import PunctuationHandler
from .grammar.myanmar_grammar import MyanmarGrammarHandler


@dataclass
class CompleteParseResult:
    """Complete parse result with every grammar feature"""
    
    # Core SVOMPTR slots
    S: Optional[str] = None
    V: Optional[str] = None
    O: Optional[str] = None
    M: Optional[str] = None
    P: Optional[str] = None
    T: Optional[str] = None
    R: Optional[str] = None
    
    # Question handling
    is_question: bool = False
    question_word: Optional[str] = None
    
    # Voice
    voice: Voice = Voice.ACTIVE
    passive_agent: Optional[str] = None
    
    # Tense
    tense: Tense = Tense.PRESENT_SIMPLE
    
    # Conditional
    is_conditional: bool = False
    conditional_type: Optional[ConditionalType] = None
    condition_clause: Optional[str] = None
    result_clause: Optional[str] = None
    
    # Reported speech
    is_reported: bool = False
    reporting_verb: Optional[str] = None
    reported_clause: Optional[str] = None
    is_indirect: bool = False
    
    # Conjunctions
    conjunctions: List[Dict] = field(default_factory=list)
    
    # Negation
    is_negated: bool = False
    negation_word: Optional[str] = None
    
    # Causative
    is_causative: bool = False
    causative_info: Optional[Dict] = None
    
    # Ellipsis
    has_ellipsis: bool = False
    omitted_words: List[str] = field(default_factory=list)
    
    # Discourse markers
    discourse_markers: List[str] = field(default_factory=list)
    
    # Emphasis
    has_emphasis: bool = False
    emphasis_type: Optional[str] = None
    
    # Prepositions
    prepositions: List[Dict] = field(default_factory=list)
    
    # Determiners
    determiners: List[Dict] = field(default_factory=list)
    
    # Numerals
    numerals: List[Dict] = field(default_factory=list)
    
    # Phrasal Verbs
    phrasal_verb: Optional[Dict] = None
    
    # Subjunctive
    subjunctive: Optional[Dict] = None
    
    # Reflexive
    reflexives: List[Dict] = field(default_factory=list)
    
    # Adverbs (Enhanced)
    adverbs: List[Dict] = field(default_factory=list)
    
    # Clauses
    clauses: List[Dict] = field(default_factory=list)
    
    # Appositives
    appositives: List[Dict] = field(default_factory=list)
    
    # Tag Questions
    tag_question: Optional[Dict] = None
    
    # Absolute Phrases
    absolute_phrase: Optional[Dict] = None
    
    # Punctuation
    punctuations: List[Dict] = field(default_factory=list)
    
    # Myanmar Grammar
    myanmar_analysis: Optional[Dict] = None
    
    # Raw
    raw_text: str = ""
    tokens: List[str] = field(default_factory=list)


class SVOMPTRCompleteParser:
    """
    Complete parser integrating ALL grammar rules
    """
    
    def __init__(self):
        # Initialize all grammar handlers
        self.voice_handler = VoiceHandler()
        self.tense_handler = TenseHandler()
        self.conditional_handler = ConditionalHandler()
        self.reported_handler = ReportedSpeechHandler()
        self.conjunction_handler = ConjunctionHandler()
        self.negation_handler = NegationHandler()
        self.causative_handler = CausativeHandler()
        self.ellipsis_handler = EllipsisHandler()
        self.discourse_handler = DiscourseHandler()
        self.emphasis_handler = EmphasisHandler()
        self.preposition_handler = PrepositionHandler()
        self.determiner_handler = DeterminerHandler()
        self.numerals_handler = NumeralsHandler()
        self.phrasal_verb_handler = PhrasalVerbHandler()
        self.subjunctive_handler = SubjunctiveHandler()
        self.reflexive_handler = ReflexiveHandler()
        self.adverbs_handler = AdverbsHandler()
        self.clause_handler = ClauseHandler()
        self.appositive_handler = AppositiveHandler()
        self.tag_question_handler = TagQuestionHandler()
        self.absolute_phrase_handler = AbsolutePhraseHandler()
        self.punctuation_handler = PunctuationHandler()
        self.myanmar_handler = MyanmarGrammarHandler()
    
    def parse(self, sentence: str) -> CompleteParseResult:
        """
        Parse sentence with all grammar rules
        """
        result = CompleteParseResult(raw_text=sentence)
        tokens = sentence.strip().split()
        result.tokens = tokens
        
        if not tokens:
            return result
        
        # Step 1: Detect and remove discourse markers
        discourse_info = self.discourse_handler.detect_discourse_markers(tokens)
        if discourse_info.has_marker:
            result.discourse_markers = discourse_info.markers
            # Remove markers for core parsing
            core_tokens = self.discourse_handler.remove_markers(tokens)
        else:
            core_tokens = tokens
        
        # Step 2: Detect reported speech
        is_reported, rep_verb, rep_clause, is_indirect = self.reported_handler.detect_reported_speech(core_tokens)
        if is_reported:
            result.is_reported = True
            result.reporting_verb = rep_verb
            result.reported_clause = rep_clause
            result.is_indirect = is_indirect
        
        # Step 3: Detect conditional
        is_cond, cond_type, cond_clause, res_clause = self.conditional_handler.detect_conditional(sentence)
        if is_cond:
            result.is_conditional = True
            result.conditional_type = cond_type
            result.condition_clause = cond_clause
            result.result_clause = res_clause
        
        # Step 4: Detect causative
        causative_info = self.causative_handler.detect_causative(core_tokens)
        if causative_info.is_causative:
            result.is_causative = True
            result.causative_info = {
                "verb": causative_info.causative_verb,
                "person": causative_info.person,
                "action": causative_info.action,
                "is_passive": causative_info.is_passive_causative
            }
        
        # Step 5: Detect negation
        has_neg, neg_word, neg_type = self.negation_handler.detect_negation(core_tokens)
        if has_neg:
            result.is_negated = True
            result.negation_word = neg_word
        
        # Step 6: Detect passive voice
        is_passive, orig_object, agent = self.voice_handler.detect_passive(core_tokens)
        if is_passive:
            result.voice = Voice.PASSIVE
            result.passive_agent = agent
        
        # Step 7: Detect tense
        tense, confidence = self.tense_handler.detect_tense(core_tokens)
        result.tense = tense
        
        # Step 8: Detect conjunctions
        conj_info_list = self.conjunction_handler.detect_conjunctions(core_tokens)
        for conj_info in conj_info_list:
            result.conjunctions.append({
                "conjunction": conj_info.conjunction,
                "left": conj_info.left_clause,
                "right": conj_info.right_clause,
                "type": conj_info.type.value
            })
        
        # Step 9: Detect prepositions
        prep_info_list = self.preposition_handler.detect_prepositions(core_tokens)
        for prep_info in prep_info_list:
            result.prepositions.append({
                "preposition": prep_info.preposition,
                "type": prep_info.type.value,
                "object": prep_info.object,
                "meaning": prep_info.meaning
            })
        
        # Step 10: Detect emphasis
        emphasis_info = self.emphasis_handler.detect_emphasis(core_tokens)
        if emphasis_info.has_emphasis:
            result.has_emphasis = True
            result.emphasis_type = emphasis_info.type.value if emphasis_info.type else None
        
        # Step 11: Detect ellipsis
        ellipsis_info = self.ellipsis_handler.detect_ellipsis(sentence)
        if ellipsis_info.has_ellipsis:
            result.has_ellipsis = True
            result.omitted_words = ellipsis_info.omitted_words
        
        # Step 12: Detect determiners
        det_info_list = self.determiner_handler.detect_determiners(core_tokens)
        for det_info in det_info_list:
            result.determiners.append({
                "word": det_info.word,
                "type": det_info.type.value,
                "modifies": det_info.modifies,
                "meaning": det_info.meaning
            })
            
        # Step 13: Detect numerals
        num_info_list = self.numerals_handler.detect_numerals(core_tokens)
        for num_info in num_info_list:
            result.numerals.append({
                "word": num_info.word,
                "type": num_info.type.value,
                "value": num_info.value,
                "myanmar": num_info.myanmar_equivalent
            })
            
        # Step 14: Detect phrasal verbs
        pv_info = self.phrasal_verb_handler.detect_phrasal_verb(core_tokens)
        if pv_info.is_phrasal:
            result.phrasal_verb = {
                "verb": pv_info.verb,
                "particle": pv_info.particle,
                "type": pv_info.type.value,
                "meaning": pv_info.meaning,
                "object": pv_info.object
            }
            
        # Step 15: Detect subjunctive
        sub_info = self.subjunctive_handler.detect_subjunctive(core_tokens)
        if sub_info.is_subjunctive:
            result.subjunctive = {
                "type": sub_info.type.value if sub_info.type else None,
                "trigger": sub_info.trigger_word,
                "meaning": sub_info.meaning
            }
            
        # Step 16: Detect reflexive
        ref_info_list = self.reflexive_handler.detect_reflexive(core_tokens)
        for ref_info in ref_info_list:
            result.reflexives.append({
                "pronoun": ref_info.pronoun,
                "type": ref_info.type.value,
                "refers_to": ref_info.refers_to,
                "meaning": ref_info.meaning
            })
            
        # Step 17: Detect adverbs
        adv_info_list = self.adverbs_handler.detect_adverbs(core_tokens)
        for adv_info in adv_info_list:
            result.adverbs.append({
                "word": adv_info.word,
                "type": adv_info.type.value,
                "meaning": adv_info.meaning,
                "intensity": adv_info.intensity
            })
            
        # Step 19: Detect clauses
        clause_info_list = self.clause_handler.detect_clauses(core_tokens)
        for c_info in clause_info_list:
            result.clauses.append({
                "type": c_info.type.value,
                "text": c_info.text,
                "subordinator": c_info.subordinator
            })
            
        # Step 20: Detect appositives
        app_info_list = self.appositive_handler.detect_appositives(core_tokens)
        for app_info in app_info_list:
            result.appositives.append({
                "main_noun": app_info.main_noun,
                "appositive": app_info.appositive,
                "is_restrictive": app_info.is_restrictive
            })
            
        # Step 21: Detect tag questions
        tag_info = self.tag_question_handler.detect_tag_question(sentence)
        if tag_info.is_tag_question:
            result.tag_question = {
                "tag": tag_info.tag,
                "auxiliary": tag_info.tag_auxiliary,
                "pronoun": tag_info.tag_pronoun
            }
            
        # Step 22: Detect absolute phrases
        abs_info = self.absolute_phrase_handler.detect_absolute_phrase(core_tokens)
        if abs_info:
            result.absolute_phrase = {
                "phrase": abs_info.phrase,
                "noun": abs_info.noun,
                "participle": abs_info.participle
            }
            
        # Step 23: Detect punctuation
        punct_info_list = self.punctuation_handler.detect_punctuation(sentence)
        for p_info in punct_info_list:
            result.punctuations.append({
                "char": p_info.char,
                "type": p_info.type.value,
                "function": p_info.function
            })
            
        # Step 24: Detect Myanmar Grammar (if sentence looks like Myanmar)
        if any(ord(c) >= 0x1000 and ord(c) <= 0x109F for c in sentence):
            result.myanmar_analysis = self.myanmar_handler.analyze_sentence(sentence)
        
        # Step 25: Core SVOMPTR parsing
        # This is where the main slot parsing happens
        self._parse_core_slots(core_tokens, result)
        
        return result
    
    def _parse_core_slots(self, tokens: List[str], result: CompleteParseResult):
        """Parse core SVOMPTR slots"""
        # If Myanmar analysis is available, use its slots
        if result.myanmar_analysis and 'slots' in result.myanmar_analysis:
            my_slots = result.myanmar_analysis['slots']
            result.S = my_slots.get('S', result.S)
            result.V = my_slots.get('V', result.V)
            result.O = my_slots.get('O', result.O)
            result.M = my_slots.get('M', result.M)
            result.P = my_slots.get('P', result.P)
            result.T = my_slots.get('T', result.T)
            result.R = my_slots.get('R', result.R)
            return

        if len(tokens) >= 1:
            result.S = tokens[0]
        if len(tokens) >= 2:
            result.V = tokens[1]
        if len(tokens) >= 3:
            # Check for question markers
            if tokens[0] in ["what", "why", "who", "when", "where", "how"]:
                result.is_question = True
                result.question_word = tokens[0]
            result.O = " ".join(tokens[2:])
        
        # Add time markers to T slot
        for i, token in enumerate(tokens):
            if token in ["today", "yesterday", "tomorrow", "now", "ဒီနေ့", "မနေ့က", "မနက်ဖြန်"]:
                result.T = token
    
    def get_summary(self, result: CompleteParseResult) -> str:
        """Get human-readable summary of parse result"""
        lines = []
        
        lines.append(f"S: {result.S}")
        lines.append(f"V: {result.V}")
        lines.append(f"O: {result.O}")
        
        if result.M:
            lines.append(f"M: {result.M}")
        if result.P:
            lines.append(f"P: {result.P}")
        if result.T:
            lines.append(f"T: {result.T}")
        if result.R:
            lines.append(f"R: {result.R}")
        
        if result.is_question:
            lines.append(f"Question: Yes (Word: {result.question_word})")
        if result.is_negated:
            lines.append(f"Negated: Yes ({result.negation_word})")
        if result.voice == Voice.PASSIVE:
            lines.append(f"Passive: Yes (Agent: {result.passive_agent})")
        if result.is_conditional:
            cond_names = {
                ConditionalType.ZERO: "Zero (General truth)",
                ConditionalType.FIRST: "First (Real possibility)",
                ConditionalType.SECOND: "Second (Unreal present)",
                ConditionalType.THIRD: "Third (Unreal past)"
            }
            lines.append(f"Conditional: {cond_names.get(result.conditional_type, 'Mixed')}")
        if result.is_reported:
            lines.append(f"Reported Speech: {result.reporting_verb} → {result.reported_clause}")
        if result.conjunctions:
            lines.append(f"Conjunctions: {len(result.conjunctions)} found")
        if result.prepositions:
            lines.append(f"Prepositions: {len(result.prepositions)} found")
        if result.determiners:
            lines.append(f"Determiners: {len(result.determiners)} found")
        if result.numerals:
            lines.append(f"Numerals: {len(result.numerals)} found")
        if result.phrasal_verb:
            lines.append(f"Phrasal Verb: {result.phrasal_verb['verb']} {result.phrasal_verb['particle']} ({result.phrasal_verb['meaning']})")
        if result.subjunctive:
            lines.append(f"Subjunctive: {result.subjunctive['trigger']} → {result.subjunctive['meaning']}")
        if result.reflexives:
            lines.append(f"Reflexives: {len(result.reflexives)} found")
        if result.adverbs:
            lines.append(f"Adverbs: {len(result.adverbs)} found")
        if result.discourse_markers:
            lines.append(f"Discourse Markers: {', '.join(result.discourse_markers)}")
        if result.has_ellipsis:
            lines.append(f"Ellipsis: Omitted {', '.join(result.omitted_words)}")
        
        return "\n".join(lines)


# Test
if __name__ == "__main__":
    parser = SVOMPTRCompleteParser()
    
    test_sentences = [
        "He said that he was tired",
        "If it rains, I will stay home",
        "The book was read by the student",
        "I have already eaten",
        "Well, actually, I don't know",
        "Never have I seen such beauty",
        "There is a cat on the roof",
        "It is raining heavily",
        "What do you want to eat?"
    ]
    
    for sentence in test_sentences:
        print(f"\n{'='*60}")
        print(f"Input: {sentence}")
        print('='*60)
        result = parser.parse(sentence)
        print(parser.get_summary(result))
