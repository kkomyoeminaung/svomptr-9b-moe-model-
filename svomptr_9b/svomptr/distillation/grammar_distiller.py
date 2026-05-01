# /svomptr_9b/svomptr/distillation/grammar_distiller.py

import json
from typing import List, Dict
import random
from ..core.svomptr_complete import SVOMPTRCompleteParser

class GrammarDistiller:
    """
    Grammar-to-Data Distiller
    Uses all 36 grammar components to generate complex training samples.
    """
    
    def __init__(self):
        self.parser = SVOMPTRCompleteParser()
        
    def generate_prompt_for_llm(self, component_name: str, count: int = 10) -> str:
        """
        LLM (Qwen/Gemma) အတွက် Grammar-specific prompt ထုတ်ပေးခြင်း
        """
        rules = self._get_rules_by_component(component_name)
        
        prompt = f"""
        Role: Expert English Teacher and Linguist
        Task: Generate {count} natural English sentences focusing on the grammar component: "{component_name}".
        
        Grammar Rules/Reference:
        {json.dumps(rules, indent=2, ensure_ascii=False)}
        
        Guidelines:
        1. Make sentences sound like real conversation or high-quality literature.
        2. Vary the length and complexity.
        3. Include Myanmar translations if possible.
        4. Output format: A JSON list of objects with "en" and "my" keys.
        
        Example Output:
        [
          {{"en": "Sentence 1", "my": "ဘာသာပြန် ၁"}},
          {{"en": "Sentence 2", "my": "ဘာသာပြန် ၂"}}
        ]
        """
        return prompt

    def _get_rules_by_component(self, component: str) -> Dict:
        """Parser ထဲက rules တွေကို ဆွဲထုတ်ခြင်း"""
        # Mapping components to handlers and their rule methods
        try:
            if component == "determiners": return self.parser.determiner_handler.get_article_rules()
            if component == "phrasal_verbs": return self.parser.phrasal_verb_handler.get_phrasal_verb_rules()
            if component == "subjunctive": return self.parser.subjunctive_handler.get_subjunctive_rules()
            if component == "clauses": return self.parser.clause_handler.get_clause_rules()
            if component == "tag_questions": return self.parser.tag_question_handler.get_tag_question_rules()
            if component == "absolute_phrases": return self.parser.absolute_phrase_handler.get_absolute_phrase_rules()
            if component == "punctuation": return self.parser.punctuation_handler.get_punctuation_rules()
            if component == "tense": return self.parser.tense_handler.get_tense_rules()
            if component == "voice": return self.parser.voice_handler.get_voice_rules()
            if component == "conditional": return self.parser.conditional_handler.get_conditional_rules()
            if component == "reported_speech": return self.parser.reported_handler.get_reported_speech_rules()
            if component == "conjunctions": return self.parser.conjunction_handler.get_conjunction_rules()
            if component == "negation": return self.parser.negation_handler.get_negation_rules()
            if component == "causative": return self.parser.causative_handler.get_causative_rules()
            if component == "ellipsis": return self.parser.ellipsis_handler.get_ellipsis_rules()
            if component == "discourse": return self.parser.discourse_handler.get_discourse_rules()
            if component == "emphasis": return self.parser.emphasis_handler.get_emphasis_rules()
            if component == "prepositions": return self.parser.preposition_handler.get_preposition_rules()
            if component == "numerals": return self.parser.numerals_handler.get_numeral_rules()
            if component == "reflexive": return self.parser.reflexive_handler.get_reflexive_rules()
            if component == "adverbs": return self.parser.adverbs_handler.get_adverb_rules()
            if component == "appositives": return self.parser.appositive_handler.get_appositive_rules()
            if component == "myanmar_grammar": 
                return {
                    "particles": self.parser.myanmar_handler.particles,
                    "tense_markers": {str(k): v for k, v in self.parser.myanmar_handler.tense_markers.items()},
                    "syntax": self.parser.myanmar_handler.myanmar_syntax_order
                }
        except AttributeError:
             return {"info": f"Explore various patterns for {component}"}
             
        return {"info": "Explore various patterns for " + component}

    def process_distilled_data(self, llm_json_response: str):
        """
        LLM က ပြန်ပေးလိုက်တဲ့ raw sentences တွေကို SVOMPTR tokens အဖြစ်ပြောင်းပြီး 
        training ready data အဖြစ် သိမ်းဆည်းခြင်း။
        """
        try:
            # Bug #14 Fix: Robust markdown and JSON cleaning
            clean_json = llm_json_response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            
            data = json.loads(clean_json)
            # Ensure it's a list
            if not isinstance(data, list):
                data = [data]
                
            processed_samples = []
            
            for item in data:
                sentence = item.get('en', item.get('sentence', ''))
                if not sentence: continue
                
                # SVOMPTR Parser နဲ့ parse လုပ်မယ်
                parse_result = self.parser.parse(sentence)
                
                # Training Pair ဖန်တီးမယ်
                sample = {
                    "input": sentence,
                    "target": self._result_to_dict(parse_result), # SVOMPTR Logical Form
                    "myanmar": item.get('my', ""),
                    "grammar_focus": self._extract_focus(parse_result)
                }
                processed_samples.append(sample)
                
            return processed_samples
        except Exception as e:
            print(f"Error processing data: {e}")
            return []

    def _result_to_dict(self, result) -> Dict:
        """Convert parse result dataclass to dict safe for JSON - Comprehensive Upgrade"""
        d = {
            "S": result.S, "V": result.V, "O": result.O,
            "M": result.M, "P": result.P, "T": result.T, "R": result.R,
            "voice": result.voice.value if hasattr(result.voice, 'value') else str(result.voice),
            "tense": result.tense.value if hasattr(result.tense, 'value') else str(result.tense),
            "is_question": result.is_question,
            "is_negated": result.is_negated,
            "is_conditional": result.is_conditional,
            "is_reported": result.is_reported,
            "is_causative": result.is_causative,
            "has_emphasis": result.has_emphasis,
            "has_ellipsis": result.has_ellipsis
        }
        
        # Lists/Complex Data
        if result.conjunctions: d["conjunctions"] = result.conjunctions
        if result.prepositions: d["prepositions"] = result.prepositions
        if result.determiners: d["determiners"] = result.determiners
        if result.numerals: d["numerals"] = result.numerals
        if result.phrasal_verb: d["phrasal_verb"] = result.phrasal_verb
        if result.subjunctive: d["subjunctive"] = result.subjunctive
        if result.reflexives: d["reflexives"] = result.reflexives
        if result.adverbs: d["adverbs"] = result.adverbs
        if result.clauses: d["clauses"] = result.clauses
        if result.appositives: d["appositives"] = result.appositives
        if result.tag_question: d["tag_question"] = result.tag_question
        if result.absolute_phrase: d["absolute_phrase"] = result.absolute_phrase
        if result.punctuations: d["punctuations"] = result.punctuations
        if result.myanmar_analysis: d["myanmar_analysis"] = result.myanmar_analysis
        
        return d

    def _extract_focus(self, result) -> List[str]:
        focus = []
        if result.phrasal_verb: focus.append("phrasal_verb")
        if result.subjunctive: focus.append("subjunctive")
        if result.is_conditional: focus.append("conditional")
        if result.is_reported: focus.append("reported_speech")
        if result.tag_question: focus.append("tag_question")
        if result.absolute_phrase: focus.append("absolute_phrase")
        if result.is_causative: focus.append("causative")
        if result.has_emphasis: focus.append("emphasis")
        if result.myanmar_analysis: focus.append("myanmar_grammar")
        if result.clauses: focus.append("clauses")
        if result.appositives: focus.append("appositives")
        return focus
