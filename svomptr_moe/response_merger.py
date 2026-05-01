class ResponseMerger:
    """
    Fuses output from Chat Expert and Domain Expert.
    Ensures technical depth + conversational fluidness.
    """
    
    def merge_logits(self, chat_logits, expert_logits, confidence):
        # Weighted average of probabilities for token-level MoE
        return (1 - confidence) * chat_logits + confidence * expert_logits

    def merge_responses(self, chat_res, expert_res, domain, confidence):
        """
        Synthesizes expert insights directly into the structural output.
        """
        if not expert_res or domain in ["chat", "general"] or confidence < 0.4:
            return chat_res
            
        # Clean up expert response from markers
        clean_expert = expert_res.replace(f"[Domain Expert: {domain.upper()}]", "").strip()
        
        # Integration logic: If chat_res has SVOMPTR labels, we append insights to the specific slots
        if "Structure: S:" in chat_res:
            header, structure = chat_res.split("Structure:", 1)
            # Add technical meta-context to the structure block
            enhanced_structure = f"{structure}\nTechnical Depth ({domain.upper()}): {clean_expert}"
            return f"{header}\n{enhanced_structure}"
            
        return f"{chat_res}\n\n[Refined by {domain.upper()} Specialist]: {clean_expert}"
