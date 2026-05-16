# /generate_data.py
import os
import sys

# Ensure project root is in path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from svomptr_9b.training.data_builder import DataBuilder

if __name__ == "__main__":
    print("📊 Starting Data Generation Wrapper...")
    raw_path = os.path.join(REPO_ROOT, "svomptr_9b", "data", "raw", "rules.json")
    out_dir = os.path.join(REPO_ROOT, "svomptr_9b", "training", "data")
    
    builder = DataBuilder(raw_path, out_dir)
    builder.build()
    print("✅ Extraction and building complete.")
