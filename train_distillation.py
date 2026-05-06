# /train_distillation.py
import os
import sys

# Critical Path Injection
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "svomptr_9b"))

from svomptr_moe.train_chat_expert import train_chat_expert

def main():
    print("🧠 Starting Distillation Training Phase...")
    # The train_chat_expert handles its own internal check for environment 
    # and resume from checkpoints.
    train_chat_expert()

if __name__ == "__main__":
    main()
