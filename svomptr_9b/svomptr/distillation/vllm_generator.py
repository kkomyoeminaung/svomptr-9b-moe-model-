# /svomptr_9b/svomptr/distillation/vllm_generator.py

import os
import json
from vllm import LLM, SamplingParams
from typing import List, Dict

class VLLMGenerator:
    """
    High-performance Synthetic Data Generator using vLLM.
    Used for Step 1 of the SVOMPTR Pipeline.
    """
    def __init__(self, model_name: str = "neural-chat-7b-v3-3", gpu_memory_utilization: float = 0.9):
        print(f"📦 Initializing vLLM Engine with: {model_name}")
        self.llm = LLM(
            model=model_name, 
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
            presence_penalty=1.1
        )

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Runs batch generation for the provided grammar prompts.
        """
        print(f"🔥 Running vLLM Batch Inference for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            
        return results

    def save_raw_responses(self, responses: List[str], output_path: str):
        """Saves raw LLM responses for debugging or manual verification."""
        with open(output_path, "w", encoding="utf-8") as f:
            for resp in responses:
                f.write(json.dumps({"raw_response": resp}, ensure_ascii=False) + "\n")
        print(f"💾 Raw responses saved to {output_path}")
