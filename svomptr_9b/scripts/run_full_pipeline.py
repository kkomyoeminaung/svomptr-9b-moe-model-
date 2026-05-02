import os
import time
import subprocess
import sys

def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=True)
    if result.returncode != 0:
        print(f"Error executing {' '.join(command)}")
        sys.exit(1)

def main():
    print("🚀 Starting SVOMPTR-9B Full Pipeline (Automated)")
    
    # 1. Generate Data
    if not os.path.exists("data/raw/distillation_output.jsonl"):                
        print("💡 Step 1: Generating synthetic data...")
        run_command(["python3", "-m", "svomptr_9b.svomptr.distillation.grammar_distiller"]) 

    # Simplified approach: Just run the main training script with phases
    print("💡 Running Full Training Pipeline...")
    run_command(["python3", "-m", "svomptr_9b.scripts.run_training"])
    
    print("✅ Full Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
