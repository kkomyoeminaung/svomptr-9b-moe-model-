import torch
import torch.nn as nn
from .config import MoEConfig
from .chat_expert import ChatExpert
from .sub_experts import SubExpert
from .domain_router import DomainRouter
from .response_merger import ResponseMerger

class ChatFirstMoE(nn.Module):
    """
    SVOMPTR-9B MoE: Chat Expert is ALWAYS ACTIVE.
    Specialized experts are activated only when router confidence exceeds threshold.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Expert 0: Main Chat Expert (2.5B)
        self.chat_expert = ChatExpert()
        
        # Experts 1-12: Sub-Experts (0.5B each)
        self.sub_experts = nn.ModuleList([
            SubExpert(domain) for domain in config.domain_names[1:]
        ])
        
        # The MoE Router (Lightweight classifier)
        self.router = DomainRouter(config.hidden_dim, config.num_experts - 1)
        
        # Response Merger
        self.merger = ResponseMerger()

    def forward(self, input_ids, attention_mask=None, query_text=""):
        # 1. Chat Expert is ALWAYS active
        chat_output = self.chat_expert(input_ids, attention_mask)
        
        # 2. Check if a domain expert is needed
        # In a real MoE, this happens per token, but for high-level domain routing,
        # we can route per-query for efficiency in this architecture.
        domain_idx, confidence = self.router.get_route(query_text)
        
        if confidence >= self.config.router_threshold:
            # 3. Activate associated Sub-Expert
            expert_output = self.sub_experts[domain_idx](input_ids, attention_mask)
            
            # 4. Hybrid Merge
            return self.merger.merge_logits(chat_output, expert_output, confidence)
        
        return chat_output

    def generate(self, query):
        """High-level inference entry point"""
        # Step 1: Chat expert generates response
        chat_gen = self.chat_expert.generate(query)
        
        # Step 2: Route query to sub-experts
        domain_idx, confidence = self.router.get_route(query)
        
        if confidence >= self.config.router_threshold:
            domain_name = self.config.domain_names[domain_idx + 1]
            print(f"[MoE] Activating Domain Expert: {domain_name} (Conf: {confidence:.2f})")
            
            # Step 3: Domain expert generates specialized technical info
            expert_gen = self.sub_experts[domain_idx].generate(query)
            
            # Step 4: Final Merged Output
            return self.merger.merge_responses(chat_gen, expert_gen, domain_name)
            
        return chat_gen
