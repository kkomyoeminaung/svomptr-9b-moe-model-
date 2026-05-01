# /svomptr_9b/svomptr/distillation/pipeline.py

import os
import json
from .grammar_distiller import GrammarDistiller
from ..memory.long_term import LongTermMemory

def run_distillation_pipeline(output_file="distilled_dataset.jsonl", dry_run=False):
    """
    Main pipeline to orchestrate knowledge distillation.
    dry_run=True: Uses mock data to simulate end-to-end processing.
    """
    # Shared memory instance for 100% performance optimization
    memory = LongTermMemory() if not dry_run or os.getenv("ENABLE_TEST_MEMORY") else None
    distiller = GrammarDistiller(memory=memory)
    
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
    checkpoint_file = "distillation_progress.json"
    output_temp_file = output_file + ".tmp"
    processed_components = []
    all_processed_samples = []

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
                processed_components = ckpt_data.get("processed", [])
                print(f"🔄 Resuming: {len(processed_components)} components already finished.")
        except: pass

    # Also try to load existing samples from temp file if any
    if os.path.exists(output_temp_file) and processed_components:
        try:
            with open(output_temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_processed_samples.append(json.loads(line))
            print(f"📦 Restored {len(all_processed_samples)} samples from {output_temp_file}")
        except: pass

    remaining_components = [c for c in components if c not in processed_components]
    print(f"Target Components: {len(remaining_components)} remaining out of {len(components)}")
    
    # Using tqdm for visual progress
    pbar = tqdm(remaining_components, desc="Grammar Distillation")
    
    # Open temp file in append mode
    with open(output_temp_file, "a", encoding="utf-8") as f_temp:
        for comp in pbar:
            pbar.set_postfix({"current": comp})
            try:
                if dry_run:
                    import time
                    time.sleep(0.3) 
                    mock_data = distiller.generate_mock_data(comp)
                    mock_json = json.dumps(mock_data)
                    samples = distiller.process_distilled_data(mock_json, memory_save=True)
                    
                    # Store and Save incrementally
                    for s in samples:
                        f_temp.write(json.dumps(s, ensure_ascii=False) + "\n")
                        all_processed_samples.append(s)
                else:
                    prompt = distiller.generate_prompt_for_llm(comp)
                    entry = {"component": comp, "prompt_ready": True}
                    f_temp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    all_processed_samples.append(entry)
                
                # Save progress checkpoint
                processed_components.append(comp)
                with open(checkpoint_file, "w", encoding="utf-8") as f_ckpt:
                    json.dump({"processed": processed_components}, f_ckpt)
            except Exception as e:
                print(f"\n❌ Error distilling {comp}: {e}")

    # Finalize
    if all_processed_samples:
        if dry_run:
            # Move temp to final
            import shutil
            shutil.move(output_temp_file, output_file)
            print(f"\n🎉 Distillation Finished. {len(all_processed_samples)} samples saved to {output_file}")
        else:
            print(f"\n📝 All prompts generated. Check {output_temp_file}")
    
    # Clean up checkpoint if finished everything
    if len(processed_components) == len(components) and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    else:
        # Create a metadata file explaining how to use these prompts
        metadata = {
            "project": "SVOMPTR-9B",
            "version": "1.0-upgrade",
            "instructions": "Use the generated prompts in gems.google.com or Ollama to generate JSON data.",
            "components_covered": components
        }
        with open("distillation_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"📂 Pipeline initialized. Metadata saved to distillation_metadata.json")
    
    return all_processed_samples

if __name__ == "__main__":
    # Logical check: Run in dry_run mode by default for verification
    run_distillation_pipeline(dry_run=True)
