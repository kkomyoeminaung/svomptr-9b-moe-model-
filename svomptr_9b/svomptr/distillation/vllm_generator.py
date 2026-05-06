import torch
import torch.nn as nn

class VLLMGenerator:
    def __init__(self, force_vllm=False):
        self.use_vllm = False
        self.llm = None
        self.pipe = None
        try:
            if force_vllm:
                from vllm import LLM
                self.llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
                self.use_vllm = True
            else:
                from transformers import pipeline as hf_pipeline
                self.pipe = hf_pipeline("text-generation",
                             model="Qwen/Qwen2.5-7B-Instruct",
                             device_map="auto")
        except Exception as e:
            print(f"Generator init failed: {e}")

    def generate_batch(self, prompts):
        if self.use_vllm:
            return self.llm.generate(prompts)
        else:
            return [res[0]['generated_text'] for res in self.pipe(prompts)]
