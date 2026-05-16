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
        force_rebuild = os.environ.get("FORCE_REBUILD", "false").lower() == "true"
        
        if os.path.exists(checkpoint_file) and not force_rebuild:
            print("🔄 Previous build progress found. Checking files...")
            files = ["phase1.jsonl", "phase2.jsonl", "phase3.jsonl", "phase4.jsonl"]
            if all((self.output_dir / f).exists() for f in files):
                print("✅ All dataset files already exist. Skipping build.")
                return
        
        if force_rebuild:
            print("♻️  FORCE_REBUILD enabled. Overwriting existing data...")

        # Bug #22 fix: Seed minimal data if everything is empty
        if not self.raw_data.get("slot_data") and not self.raw_data.get("unlabeled_data"):
            print("💡 No raw data seeds found. Injecting minimal synthetic seed data...")
            self.raw_data["unlabeled_data"] = [
                {"text": "The cat sat on the mat."},
                {"text": "Artificial Intelligence is transforming the world."},
                {"text": "Myanmar language has beautiful structural nuances."}
            ]
            self.raw_data["causal_data"] = [
                {"input": "Start: I am here.", "output": "Reasoning: Presence established."}
            ]
            self.raw_data["slot_data"] = [
                {"input": "He runs.", "target": "S:He, V:runs"}
            ]

        # Add special cases for grammar logic
        self.add_virtual_subject_training_data()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Import and merge distilled samples if they exist (Integrity Check)
        # Fix: Find path relative to the current project structure
        try:
            root_dir = Path(__file__).parent.parent.parent
            distilled_path = root_dir / "data" / "raw" / "rules.json"
        except:
            distilled_path = Path("data/raw/rules.json")
            
        if distilled_path.exists():
            print(f"📦 Integrating real Distilled Data from Step 1: {distilled_path}")
            try:
                with open(distilled_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                if content.startswith('{') or content.startswith('['):
                    # Standard JSON format
                    raw_json = json.loads(content)
                    if isinstance(raw_json, dict):
                        for key in ['slot_data', 'chat_data', 'unlabeled_data', 'causal_data']:
                            if key in raw_json:
                                self.raw_data.setdefault(key, []).extend(raw_json[key])
                    elif isinstance(raw_json, list):
                        for item in raw_json:
                            if "en" in item and "my" in item:
                                self.raw_data.setdefault("slot_data", []).append({"input": item["en"], "target": f"S:{item.get('component', 'unknown')}"})
                                self.raw_data.setdefault("chat_data", []).append({"instruction": f"Translate and analyze: {item['en']}", "input": "", "output": f"Burmese: {item['my']}\nComponent: {item.get('component', 'N/A')}"})
                else:
                    # JSONL format
                    for line in content.split('\n'):
                        if line.strip():
                            item = json.loads(line)
                            if "en" in item and "my" in item:
                                self.raw_data.setdefault("slot_data", []).append({"input": item["en"], "target": f"S:{item.get('component', 'unknown')}"})
                                self.raw_data.setdefault("chat_data", []).append({"instruction": f"Translate and analyze: {item['en']}", "input": "", "output": f"Burmese: {item['my']}\nComponent: {item.get('component', 'N/A')}"})
            except Exception as e:
                print(f"⚠️ Error merging distilled data: {e}")

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
                # Ensure unified format: {"en": ..., "target": ...}
                if isinstance(item, dict):
                    if "input" in item and "target" in item:
                        unified = {"en": item["input"], "target": item["target"]}
                    elif "en" in item and "target" in item:
                        unified = item
                    elif "text" in item:
                        unified = {"text": item["text"]}
                    else:
                        unified = item
                else:
                    unified = {"text": str(item)}
                f.write(json.dumps(unified, ensure_ascii=False) + "\n")
        print(f"✅ Created {filename} with {len(data)} samples.")
