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
    
    def __init__(self, memory=None):
        self.parser = SVOMPTRCompleteParser()
        self.memory = memory
        
    def generate_prompt_for_llm(self, component_name: str, count: int = 10) -> str:
        """
        Enhances the distillation prompt with strict formatting and Burmese nuance instructions.
        """
        rules = self._get_rules_by_component(component_name)
        
        prompt = f"""
        Role: Senior Linguistics Expert (English-Myanmar Specialist)
        Task: Generate {count} diverse sentences focusing on the grammar component: "{component_name}".
        
        Grammar Rules/Reference:
        {json.dumps(rules, indent=2, ensure_ascii=False)}
        
        Requirements:
        1. Context: Real-world conversation, academic, or professional usage.
        2. Bilingual: Provide the English sentence and its natural Myanmar (Burmese) translation.
        3. Burmese Nuances: Use correct particles (e.g., -တယ်, -နေတယ်, -ခဲ့တယ်) and honorifics where appropriate.
        4. No explanation: Just the JSON data.
        5. Format: Return ONLY a valid JSON list of objects:
        [
          {{"en": "...", "my": "...", "sentence": "...", "component": "{component_name}"}},
          ...
        ]
        """
        return prompt

    def generate_mock_data(self, component: str):
        """Generates realistic mock data for testing distillation pipeline without an external LLM."""
        mocks = {
            "tense": [
                {"en": "I go to school every day.", "my": "ကျွန်တော် နေ့တိုင်း ကျောင်းသွားတယ်။"},
                {"en": "She is reading a book now.", "my": "သူမ အခု စာအုပ်ဖတ်နေတယ်။"},
                {"en": "They have finished their work.", "my": "သူတို့ သူတို့ရဲ့ အလုပ်ကို ပြီးစီးခဲ့ပြီ။"}
            ],
            "myanmar_particles": [
                {"en": "Did you eat?", "my": "မင်း စားပြီးပြီလား။"},
                {"en": "I want to eat.", "my": "ကျွန်တော် စားချင်တယ်။"},
                {"en": "Please come here.", "my": "ဒီကို လာခဲ့ပါ။"}
            ]
        }
        # Default fallback
        default = [{"en": f"Sample for {component}", "my": f"{component} အတွက် နမူနာ"}]
        return mocks.get(component, default)

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

    def process_distilled_data(self, llm_json_response: str, memory_save=False):
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
                
                # Save to Long Term Memory for RAG and Dreaming
                if self.memory and memory_save:
                    self.memory.store_memory(
                        sentence, 
                        f"SVOMPTR: {json.dumps(sample['target'])} | MM: {sample['myanmar']}"
                    )
                
            return processed_samples
        except Exception as e:
            print(f"Error processing data: {e}")
            return []

    def export_dataset(self, samples: List[Dict], filename: str = "distilled_finetuning.jsonl"):
        """Save processed samples to a JSONL file for training."""
        import os
        os.makedirs("data/distilled", exist_ok=True)
        path = os.path.join("data/distilled", filename)
        with open(path, "a", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"✅ Exported {len(samples)} samples to {path}")

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
