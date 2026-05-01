# /svomptr_9b/training/data_builder.py

import json
import os
import random
from pathlib import Path
from tqdm import tqdm

class DataBuilder:
    """Builds 4-phase training data with high-quality bilingual reasoning (CoT) pairs"""
    def __init__(self, raw_data_path: str, output_dir: str):
        if os.path.exists(raw_data_path):
            with open(raw_data_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
        else:
            self.raw_data = {
                "slot_data": [],
                "unlabeled_data": [],
                "causal_data": [],
                "chat_data": []
            }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def add_virtual_subject_training_data(self):
        """Add critical training examples for virtual subjects, pro-drop, and advanced SVOMPTR rules"""
        print("💡 Adding Virtual Subject and Advanced Syntax logic samples...")
        
        virtual_examples = [
            {"en": "I am eating.", "my": "စားနေတယ်။", "svomptr": "S(hidden):I, V:eating"},
            {"en": "It is raining.", "my": "မိုးရွာနေတယ်။", "svomptr": "S(hidden):It, V:raining"},
            {
                "instruction": "Parse: There is a cat",
                "input": "",
                "output": "S:<VIRTUAL_EXISTENTIAL> V:is O:a cat\nInterpretation: Introducing new subject 'cat'"
            },
            {
                "instruction": "Parse: Sit down",
                "input": "",
                "output": "S:You(implied) V:sit M:down\nInterpretation: Command with implied subject"
            }
        ]
        
        if "chat_data" not in self.raw_data: self.raw_data["chat_data"] = []
        self.raw_data["chat_data"].extend(virtual_examples)
        
        if "slot_data" not in self.raw_data: self.raw_data["slot_data"] = []
        for ex in virtual_examples:
            if "svomptr" in ex:
                self.raw_data["slot_data"].append({"input": ex["en"], "target": ex["svomptr"]})

    def build(self):
        print("🚀 Building Enhanced 4-phase training dataset...")
        
        # Check if data already exists to avoid redundant building
        checkpoint_file = self.output_dir / ".builder_progress"
        if os.path.exists(checkpoint_file):
            print("🔄 Previous build progress found. Checking files...")
            files = ["phase1.jsonl", "phase2.jsonl", "phase3.jsonl", "phase4.jsonl"]
            if all((self.output_dir / f).exists() for f in files):
                print("✅ All dataset files already exist. Skipping build.")
                return

        # Add special cases for grammar logic
        self.add_virtual_subject_training_data()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Phase 1: Slot Prediction
        self._write_jsonl("phase1.jsonl", self.raw_data.get("slot_data", []))
        
        # Phase 2: MLM (Language Pretraining)
        self._write_jsonl("phase2.jsonl", self.raw_data.get("unlabeled_data", []))
        
        # Phase 3: Causal (Reasoning + Bilingual Contrastive)
        self._write_jsonl("phase3.jsonl", self.raw_data.get("causal_data", []))
        
        # Phase 4: Conversation (Bilingual SFT + CoT)
        self._write_jsonl("phase4.jsonl", self.raw_data.get("chat_data", []))
        
        # Mark as finished
        with open(checkpoint_file, "w") as f:
            f.write("finished")
            
        print(f"✅ Data built in {self.output_dir}. Total Samples: {len(self.raw_data.get('chat_data', []))}")

    def _write_jsonl(self, filename, data):
        if not data:
            print(f"⚠️ Warning: No data for {filename}")
            return
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for item in tqdm(data, desc=f"Writing {filename}"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Created {filename} with {len(data)} samples.")
