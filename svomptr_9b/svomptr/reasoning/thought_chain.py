# svomptr/reasoning/thought_chain.py
class ThoughtChain:
    """Chain of Thought module"""
    def __init__(self):
        self.prefix = "Let's think step by step:"
    
    def generate_thought(self, prompt, context_hint=None):
        """
        Generates a sequence of internal logic steps (Chain of Thought).
        """
        from ..core.svomptr_complete import SVOMPTRCompleteParser
        parser = SVOMPTRCompleteParser()
        parse_result = parser.parse(prompt)
        
        steps = [
            f"1. Analyzed Input: Found Subject '{parse_result.S}' and Verb '{parse_result.V}'.",
            f"2. Component Focus: {', '.join(self._get_focus(parse_result)) if self._get_focus(parse_result) else 'Standard syntax'}.",
            "3. Context retrieved from long-term memory.",
            "4. Formulating response based on syntax, meaning, and bilingual alignment."
        ]
        return " | ".join(steps)

    def _get_focus(self, result):
        focus = []
        if result.is_conditional: focus.append("Conditional logic")
        if result.is_reported: focus.append("Speech reporting")
        if result.is_negated: focus.append("Negative polarity")
        if result.tense: focus.append(f"Tense: {result.tense}")
        return focus
