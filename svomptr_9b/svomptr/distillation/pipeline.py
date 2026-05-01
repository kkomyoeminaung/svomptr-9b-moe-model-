# /svomptr_9b/svomptr/distillation/pipeline.py

import os
import json
from .grammar_distiller import GrammarDistiller

def run_distillation_pipeline(output_file="distilled_dataset.jsonl"):
    """
    Main pipeline to orchestrate knowledge distillation.
    In a real scenario, this would connect to an LLM API.
    """
    distiller = GrammarDistiller()
    
    # Bug #13 Fix: Updated list to match "36 components" (approx) 
    # Adding more granular categories for comprehensive coverage
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
    
    # Ensure directory for output exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting SVOMPTR Knowledge Distillation Pipeline...")
    print(f"Target Components: {len(components)}")
    
    results_summary = []

    # Note: In this environment, we don't have direct LLM API access here,
    # so we provide the prompt generation logic and the processing logic.
    # The user can run this in Colab using Gemini/Ollama.
    
    for comp in components:
        prompt = distiller.generate_prompt_for_llm(comp)
        results_summary.append({
            "component": comp,
            "status": "ready_for_llm",
            "prompt_length": len(prompt)
        })
        print(f"Generated prompt for: {comp}")

    # Create a metadata file explaining how to use these prompts
    metadata = {
        "project": "SVOMPTR-9B",
        "version": "1.0-upgrade",
        "instructions": "Use the generated prompts in gems.google.com or Ollama to generate JSON data, then feed back to process_distilled_data.",
        "components_covered": components
    }
    
    with open("distillation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Pipeline initialized. Metadata saved to distillation_metadata.json")
    return results_summary

if __name__ == "__main__":
    run_distillation_pipeline()
