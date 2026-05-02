# /svomptr_9b/svomptr/distillation/pipeline.py

import os
import json
import time
from .grammar_distiller import GrammarDistiller
from .vllm_generator import VLLMGenerator
from ..memory.long_term import LongTermMemory

def run_distillation_pipeline(output_file="distilled_dataset.jsonl", dry_run=False, use_vllm=False):
    """
    Main pipeline to orchestrate knowledge distillation.
    dry_run=True: Uses mock rule-based logic.
    use_vllm=True: Uses vLLM to generate high-quality synthetic data.
    """
    memory = LongTermMemory() if not dry_run or os.getenv("ENABLE_TEST_MEMORY") else None
    distiller = GrammarDistiller(memory=memory)
    vllm_gen = None
    
    if use_vllm and not dry_run:
        try:
            vllm_gen = VLLMGenerator()
        except Exception as e:
            print(f"⚠️ vLLM failed to initialize: {e}. Falling back to prompt-only mode.")
            use_vllm = False
    
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
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    from tqdm import tqdm
    print(f"🚀 SVOMPTR Knowledge Distillation Pipeline {'(DRY RUN)' if dry_run else ''}")
    
    # Checkpoint support
    checkpoint_file = os.path.join(output_dir, "distillation_status.json") if output_dir else "distillation_status.json"
    output_temp_file = output_file + ".tmp"
    processed_components = []
    all_processed_samples = []

    # 1. Environment Health Check
    if output_dir:
        try:
            test_file = os.path.join(output_dir, ".write_test")
            with open(test_file, "w") as f: f.write("ok")
            os.remove(test_file)
            print("✅ Storage access verified.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Cannot write to {output_dir}. Please check your Drive mount!")
            return

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
                processed_components = ckpt_data.get("processed", [])
                print(f"🔄 Resuming: {len(processed_components)} components already finished.")
        except: pass

    # Restore existing samples
    if os.path.exists(output_temp_file):
        try:
            with open(output_temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_processed_samples.append(json.loads(line))
            print(f"📦 Restored {len(all_processed_samples)} samples from temp file.")
        except: pass

    remaining_components = [c for c in components if c not in processed_components]
    if not remaining_components:
        print("✅ Pipeline already completed for all components.")
        return

    print(f"Target Components: {len(remaining_components)} remaining out of {len(components)}")
    
    pbar = tqdm(remaining_components, desc="Distillation Progress")
    
    # Use unbuffered writing (flushing) to ensure Drive gets the data immediately
    with open(output_temp_file, "a", encoding="utf-8", buffering=1) as f_temp:
        for comp in pbar:
            pbar.set_postfix({"current": comp})
            try:
                if dry_run:
                    import time
                    # Use the new rule-based synthetic generator (Step 1 Real Logic)
                    samples_raw = distiller.generate_synthetic_data(comp)
                    mock_json = json.dumps(samples_raw)
                    samples = distiller.process_distilled_data(mock_json, memory_save=True)
                    
                    for i, s in enumerate(samples):
                        f_temp.write(json.dumps(s, ensure_ascii=False) + "\n")
                        if i % 10 == 0:
                            f_temp.flush()
                            try:
                                os.fsync(f_temp.fileno())
                            except: pass
                        all_processed_samples.append(s)
                elif use_vllm and vllm_gen:
                    print(f"📡 Requesting vLLM to generate data for: {comp}")
                    prompt = distiller.generate_prompt_for_llm(comp, count=20) # Generate 20 samples per component
                    llm_outputs = vllm_gen.generate_batch([prompt])
                    
                    for output_text in llm_outputs:
                        samples = distiller.process_distilled_data(output_text, memory_save=True)
                        for s in samples:
                            f_temp.write(json.dumps(s, ensure_ascii=False) + "\n")
                            all_processed_samples.append(s)
                    
                    f_temp.flush()
                    try:
                        os.fsync(f_temp.fileno())
                    except: pass
                else:
                    prompt = distiller.generate_prompt_for_llm(comp)
                    entry = {"component": comp, "prompt_ready": True, "timestamp": time.time()}
                    f_temp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f_temp.flush()
                    try:
                        os.fsync(f_temp.fileno())
                    except: pass
                    all_processed_samples.append(entry)
                
                # Checkpoint persistence
                processed_components.append(comp)
                with open(checkpoint_file, "w", encoding="utf-8") as f_ckpt:
                    json.dump({"processed": processed_components, "total_samples": len(all_processed_samples)}, f_ckpt)
            
            except Exception as e:
                print(f"\n🛑 Error at component {comp}: {e}")
                print("⚠️ Stopping pipeline to prevent data corruption. Please fix and restart.")
                return # Stop immediately on error

    # Finalize only if finished
    if len(processed_components) == len(components):
        import shutil
        try:
            # Copy first then delete temp as Drive move can be unstable
            shutil.copy2(output_temp_file, output_file)
            if os.path.exists(checkpoint_file): os.remove(checkpoint_file)
            if os.path.exists(output_temp_file): os.remove(output_temp_file)
            print(f"\n🎉 FULLY FINISHED. {len(all_processed_samples)} samples synced to {output_file}")
        except Exception as e:
            print(f"⚠️ Error finalizing file: {e}. Data is safe in {output_temp_file}")
    else:
        print(f"\n⚠️ Pipeline interrupted. Progress saved in {output_temp_file}")

    # Create a metadata file
    metadata = {
        "project": "SVOMPTR-9B",
        "version": "1.0-upgrade",
        "instructions": "Use the generated prompts in gems.google.com or Ollama to generate JSON data.",
        "components_covered": components
    }
    with open("distillation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"📂 Pipeline status updated in distillation_metadata.json")
    
    return all_processed_samples

if __name__ == "__main__":
    # Logical check: Run in dry_run mode by default for verification
    run_distillation_pipeline(dry_run=True)
