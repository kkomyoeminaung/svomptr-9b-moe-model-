# /svomptr_9b/svomptr/distillation/vllm_generator.py

import os
import json
import torch
from typing import List, Dict

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class VLLMGenerator:
    """
    High-performance Synthetic Data Generator.
    Supports vLLM (GPU) with graceful fallback to Transformers (CPU/GPU).
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization: float = 0.9):
        self.use_vllm = VLLM_AVAILABLE and torch.cuda.is_available()
        self.model_name = model_name
        
        if self.use_vllm:
            print(f"🚀 Initializing vLLM Engine (GPU) with: {model_name}")
            try:
                self.llm = LLM(
                    model=model_name, 
                    gpu_memory_utilization=gpu_memory_utilization,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    max_model_len=4096
                )
                self.sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=1024,
                    presence_penalty=1.1
                )
            except Exception as e:
                print(f"⚠️ vLLM Init Failed: {e}. Falling back to Transformers Mode.")
                self.use_vllm = False
        
        if not self.use_vllm:
            print(f"🐢 vLLM not available or no GPU found. Using Transformers mode.")
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline(
                "text-generation", 
                model=model_name, 
                device=device,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Runs batch generation for the provided grammar prompts.
        """
        if self.use_vllm:
            print(f"🔥 Running vLLM Batch Inference for {len(prompts)} prompts...")
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [output.outputs[0].text for output in outputs]
        else:
            print(f"🔄 Running Transformers Inference for {len(prompts)} prompts...")
            results = []
            for prompt in prompts:
                out = self.pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)
                # Extract only the generated part
                gen_text = out[0]['generated_text']
                if gen_text.startswith(prompt):
                    gen_text = gen_text[len(prompt):]
                results.append(gen_text.strip())
            return results

    def save_raw_responses(self, responses: List[str], output_path: str):
        """Saves raw LLM responses for debugging or manual verification."""
        with open(output_path, "w", encoding="utf-8") as f:
            for resp in responses:
                f.write(json.dumps({"raw_response": resp}, ensure_ascii=False) + "\n")
        print(f"💾 Raw responses saved to {output_path}")
