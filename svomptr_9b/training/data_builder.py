# svomptr_9b/training/data_builder.py

import json
import os
import random
from pathlib import Path
from tqdm import tqdm

class DataBuilder:
    """Builds 4-phase training data with high-quality bilingual reasoning (CoT) pairs"""
    def __init__(self, raw_data_path: str, output_dir: str):
        self.raw_data = json.load(open(raw_data_path, 'r')) # Expected to have CoT & Bilingual data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def build(self):
        print("🚀 Building Enhanced 4-phase training dataset...")
        
        # Phase 1: Slot Prediction
        self._build_phase1()
        
        # Phase 2: MLM (Language Pretraining)
        self._build_phase2()
        
        # Phase 3: Causal (Reasoning + Bilingual Contrastive)
        self._build_phase3()
        
        # Phase 4: Conversation (Bilingual SFT + CoT)
        self._build_phase4()
        
        print("✅ Bilingual Reasoning Data built successfully.")

    def _build_phase1(self):
        print("Building Phase 1 (Slot Prediction)...")
        with open(self.output_dir / "phase1.jsonl", "w") as f:
            for item in tqdm(self.raw_data["slot_data"]):
                f.write(json.dumps(item) + "\n")
                
    def _build_phase2(self):
        print("Building Phase 2 (Multilingual Masked LM)...")
        with open(self.output_dir / "phase2.jsonl", "w") as f:
            for item in tqdm(self.raw_data["unlabeled_data"]):
                # Ensure balance of Myanmar/English here if raw_data is mixed
                f.write(json.dumps(item) + "\n")

    def _build_phase3(self):
        print("Building Phase 3 (Bilingual Causal Reasoning Data)...")
        with open(self.output_dir / "phase3.jsonl", "w") as f:
            for item in tqdm(self.raw_data["causal_data"]):
                # Ensure reasoning pairs look like: {"input": "...", "reasoning": "...", "output": "..."}
                f.write(json.dumps(item) + "\n")

    def _build_phase4(self):
        print("Building Phase 4 (Bilingual Conversation with CoT)...")
        with open(self.output_dir / "phase4.jsonl", "w") as f:
            for item in tqdm(self.raw_data["chat_data"]):
                # Apply CoT template if not already present
                f.write(json.dumps(item) + "\n")

    def add_virtual_subject_training_data(self):
        """Add critical training examples for virtual subjects and advanced SVOMPTR rules"""
        
        virtual_examples = [
            # ========================================
            # There is (Existential) examples
            # ========================================
            {
                "instruction": "Parse: There is a cat",
                "input": "",
                "output": "S:<VIRTUAL_EXISTENTIAL> V:is O:a cat\nInterpretation: Introducing new subject 'cat'"
            },
            # ... existing existential examples ...
            
            # ========================================
            # Imperative (Implied Subject)
            # ========================================
            {
                "instruction": "Parse: Sit down",
                "input": "",
                "output": "S:You(implied) V:sit M:down\nInterpretation: Command with implied subject"
            },
            {
                "instruction": "Parse: Beat him",
                "input": "",
                "output": "S:You(implied) V:beat O:him\nInterpretation: Command with object"
            },
            
            # ========================================
            # Interjections (Emotion in M)
            # ========================================
            {
                "instruction": "Parse: Wow!",
                "input": "",
                "output": "M:wow\nInterpretation: Emotional interjection"
            },
            {
                "instruction": "Parse: Oh!",
                "input": "",
                "output": "M:oh\nInterpretation: Emotional interjection"
            },
            
            # ========================================
            # Intransitive Verbs (No Object)
            # ========================================
            {
                "instruction": "Parse: He sleeps",
                "input": "",
                "output": "S:He V:sleeps\nInterpretation: Intransitive action"
            },
            
            # ========================================
            # To-infinitive (Reason in R)
            # ========================================
            {
                "instruction": "Parse: I want to eat",
                "input": "",
                "output": "S:I V:want R:to eat\nInterpretation: Action with purpose"
            },
            {
                "instruction": "Parse: He needs to go",
                "input": "",
                "output": "S:He V:needs R:to go\nInterpretation: Condition with goal"
            },
            
            # ========================================
            # Gerunds and Manner -ing
            # ========================================
            {
                "instruction": "Parse: Running is good",
                "input": "",
                "output": "S:Running V:is O:good\nInterpretation: Gerund as subject"
            },
            {
                "instruction": "Parse: He came running",
                "input": "",
                "output": "S:He V:came M:running\nInterpretation: Participle as manner"
            },
            
            # ========================================
            # Questions and Modals
            # ========================================
            {
                "instruction": "Parse: What do you want?",
                "input": "",
                "output": "Q:What AUX:do S:you V:want\nInterpretation: Wh-question"
            },
            {
                "instruction": "Parse: May I come in?",
                "input": "",
                "output": "Q:May S:I V:come M:in\nInterpretation: Permission request"
            },
            
            # ========================================
            # Voice (Active/Passive)
            # ========================================
            {
                "instruction": "Parse: The cake was eaten by John",
                "input": "",
                "output": "S:The cake V:was eaten AGENT:by John\nInterpretation: Passive Voice"
            },
            
            # ========================================
            # Conditionals
            # ========================================
            {
                "instruction": "Parse: If it rains, I will stay",
                "input": "",
                "output": "COND:If it rains S:I V:will stay\nInterpretation: First Conditional"
            },
            
            # ========================================
            # Reported Speech
            # ========================================
            {
                "instruction": "Parse: He said that he was happy",
                "input": "",
                "output": "S:He V:said CLAUSE:that he was happy\nInterpretation: Indirect Speech"
            },
            
            # ========================================
            # Noun Clauses (S/O)
            # ========================================
            {
                "instruction": "Parse: What he said is true",
                "input": "",
                "output": "S:[What he said] V:is O:true\nInterpretation: Wh-clause as subject"
            }
        ]
        
        # Add to training dataset
        if "chat_data" in self.raw_data:
            self.raw_data["chat_data"].extend(virtual_examples)
        else:
            self.raw_data["chat_data"] = virtual_examples
