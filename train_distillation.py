# /train_distillation.py
import os
import sys

# Ensure project root is in path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from svomptr_9b.scripts.distillation_pipeline import run_distillation

if __name__ == "__main__":
    print("🌟 Starting Distillation Training Wrapper...")
    run_distillation()
