# svomptr/core/tokenizer.py

from transformers import AutoTokenizer
import pyidaungsu as pds
from typing import List, Dict

class SVOMPTRTokenizer:
    """Multilingual Tokenizer (Myanmar + English)"""
    def __init__(self, config):
        self.config = config
        # Load a base multilingual tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        self.eos_id = self.tokenizer.eos_token_id

        # Virtual subject special tokens
        self.virtual_tokens = {
            "<VIRTUAL_EXISTENTIAL>": self._add_token("<VIRTUAL_EXISTENTIAL>"),
            "<VIRTUAL_WEATHER>": self._add_token("<VIRTUAL_WEATHER>"),
            "<VIRTUAL_GENERAL>": self._add_token("<VIRTUAL_GENERAL>"),
        }
        
    def _add_token(self, token: str) -> int:
        """Add a new token to vocabulary"""
        if token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        return self.tokenizer.convert_tokens_to_ids(token)
        
    def encode(self, text: str) -> List[int]:
        # Pre-process Myanmar text
        text = pds.tokenize(text, lang='mm', form='word')
        text = " ".join(text)
        return self.tokenizer.encode(text)
        
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def save(self, path: str):
        self.tokenizer.save_pretrained(path)
