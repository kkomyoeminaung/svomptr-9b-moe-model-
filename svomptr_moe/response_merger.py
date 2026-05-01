class ResponseMerger:
    """
    Fuses output from Chat Expert and Domain Expert.
    Ensures technical depth + conversational fluidness.
    """
    
    def merge_logits(self, chat_logits, expert_logits, confidence):
        # Weighted average of probabilities for token-level MoE
        return (1 - confidence) * chat_logits + confidence * expert_logits

    def merge_responses(self, chat_res, expert_res, domain):
        """
        Human-readable merger for final generation strings.
        Clean professional merge without raw system prefixes.
        """
        if domain == "chat" or domain == "general":
            return chat_res
            
        return f"[{domain.capitalize()} expertise applied]\n\n{expert_res}\n\n{chat_res}"
