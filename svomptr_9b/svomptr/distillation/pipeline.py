# /svomptr_9b/svomptr/distillation/pipeline.py

import os
import json
import time
from datetime import datetime
from .grammar_distiller import GrammarDistiller
from .vllm_generator import VLLMGenerator
from ..memory.long_term import LongTermMemory

def run_distillation_pipeline(output_file="distilled_dataset.jsonl", dry_run=False, use_vllm=False, target_total_samples=5000000):
    """
    Main pipeline to orchestrate knowledge distillation.
    target_total_samples: Target number of samples to reach.
    """
    memory = LongTermMemory() if not dry_run or os.getenv("ENABLE_TEST_MEMORY") else None
    distiller = GrammarDistiller(memory=memory)
    vllm_gen = None
    try:
        # We always try to initialize VLLMGenerator if not explicitly doing a dry_run
        # It internally handles the fallback from vLLM (GPU) to Transformers (CPU/GPU)
        if not dry_run:
            vllm_gen = VLLMGenerator()
    except Exception as e:
        print(f"⚠️ Generator initialization failed: {e}. Falling back to rule-based generation.")
    
    actual_use_gpu = vllm_gen.use_vllm if vllm_gen else False
    
    components = [
        "tense", "voice", "conditional", "reported_speech", 
        "conjunctions", "negation", "causative", "ellipsis", 
        "discourse", "emphasis", "prepositions", "determiners", 
        "numerals", "phrasal_verbs", "subjunctive", "reflexive", 
        "adverbs", "clauses", "appositives", "tag_questions", 
        "absolute_phrases", "punctuation", "myanmar_particles",
        "honorifics", "verb_suffixes", "noun_markers", "demonstratives",
        "relative_clauses", "comparative_degree", "superlative",
        "idiomatic_expressions", "interjections", "modal_verbs",
        "gerunds", "infinitives", "participles"
    ]
    
    output_temp_file = output_file + ".tmp"
    checkpoint_file = output_file + ".ckpt.json"
    
    # 1. Load Progress
    progress = {"total_samples": 0, "component_index": 0}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                progress = json.load(f)
                print(f"🔄 Resuming from checkpoint: {progress['total_samples']} samples collected.")
        except: pass

    # 2. Main Loop
    current_samples = progress['total_samples']
    comp_idx = progress['component_index']
    
    from tqdm import tqdm
    pbar = tqdm(total=target_total_samples, initial=current_samples, desc="🚀 SVOMPTR Synthetic Generation")
    
    with open(output_temp_file, "a", encoding="utf-8", buffering=1) as f_out:
        while current_samples < target_total_samples:
            comp = components[comp_idx % len(components)]
            pbar.set_description(f"📉 Generating: {comp}")
            
            # Generate a batch - Dynamically scale based on hardware
            if actual_use_gpu:
                num_prompts = 5
                batch_size_per_prompt = 10
            elif vllm_gen: # CPU Transformers mode
                num_prompts = 1
                batch_size_per_prompt = 5
            else: # Rule-based / dry_run
                num_prompts = 1
                batch_size_per_prompt = 5
            
            try:
                samples = []
                if vllm_gen:
                    # High quality LLM generation (GPU vLLM or CPU Transformers)
                    prompts = [distiller.generate_prompt_for_llm(comp, count=batch_size_per_prompt) for _ in range(num_prompts)]
                    outputs = vllm_gen.generate_batch(prompts)
                    for out in outputs:
                        samples.extend(distiller.process_distilled_data(out, memory_save=True))
                else:
                    # Rule-based synthetic generation
                    samples = distiller.generate_synthetic_data(comp, count=batch_size_per_prompt)
                    samples = distiller.process_distilled_data(json.dumps(samples), memory_save=True)

                if not samples:
                    print(f"⚠️ No samples generated for {comp}. Skipping...")
                    comp_idx += 1
                    continue

                # Save samples
                for s in samples:
                    f_out.write(json.dumps(s, ensure_ascii=False) + "\n")
                    current_samples += 1
                
                f_out.flush()
                try: os.fsync(f_out.fileno())
                except: pass
                
                # Update progress
                comp_idx += 1
                pbar.update(len(samples))
                
                # Save Checkpoint
                with open(checkpoint_file, "w") as f_ckpt:
                    json.dump({
                        "total_samples": current_samples,
                        "component_index": comp_idx,
                        "timestamp": time.time()
                    }, f_ckpt)
                
                # Check for session timeout prevention (Colab)
                if current_samples % 500 < len(samples):
                     print(f"💓 [HEARTBEAT] {datetime.now().strftime('%H:%M:%S')} | Progress: {current_samples}/{target_total_samples}")

            except Exception as e:
                print(f"🛑 Error during generation: {e}")
                break

    # 3. Finalize
    if current_samples >= target_total_samples:
        import shutil
        shutil.copy2(output_temp_file, output_file)
        print(f"✅ Target reached! Dataset saved to {output_file}")
    else:
        print(f"⚠️ Pipeline session ended at {current_samples} samples. Data is safe in {output_temp_file}. Run again to resume.")
    
    return current_samples

if __name__ == "__main__":
    # Logical check: Run in dry_run mode by default for verification
    run_distillation_pipeline(dry_run=True)
