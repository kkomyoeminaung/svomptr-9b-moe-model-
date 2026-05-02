# /train_distillation.py
import os
import sys

# Critical Path Injection
PROJECT_ROOT = "/content/svomptr-project"
ALTERNATE_ROOT = "/content/svomptr_9b"
for p in [PROJECT_ROOT, ALTERNATE_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from svomptr_moe.train_chat_expert import train_chat_expert

def main():
    print("🧠 Starting Distillation Training Phase...")
    # The train_chat_expert handles its own internal check for environment 
    # and resume from checkpoints.
    train_chat_expert()

if __name__ == "__main__":
    main()
