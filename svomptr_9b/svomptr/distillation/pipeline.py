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
    
    print(f"🚀 SVOMPTR Knowledge Distillation Pipeline {'(DRY RUN)' if dry_run else ''}")
    print(f"Target Components: {len(components)}")
    
    all_processed_samples = []

    for comp in components:
        if dry_run:
            mock_data = distiller.generate_mock_data(comp)
            mock_json = json.dumps(mock_data)
            samples = distiller.process_distilled_data(mock_json, memory_save=True)
            all_processed_samples.extend(samples)
            print(f"✅ Processed {len(samples)} mock samples for: {comp}")
        else:
            prompt = distiller.generate_prompt_for_llm(comp)
            print(f"📝 Prompt generated for: {comp}")

    if dry_run:
         distiller.export_dataset(all_processed_samples, output_file)
         print(f"🎉 Dry run complete. {len(all_processed_samples)} samples saved to {output_file}")
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
