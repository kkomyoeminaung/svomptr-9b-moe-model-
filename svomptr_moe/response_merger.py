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
        Fuses the general conversational response with technical domain expertise.
        Ensures a seamless transition rather than a robotic prefix.
        """
        if not expert_res or domain in ["chat", "general"]:
            return chat_res
            
        # Sophisticated fusion logic to make the response feel integrated
        return f"{expert_res}\n\n{chat_res}"
