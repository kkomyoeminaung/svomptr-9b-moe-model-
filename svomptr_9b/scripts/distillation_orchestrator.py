# /svomptr_9b/scripts/distillation_orchestrator.py

import os
import sys
import subprocess
import time
import threading
from datetime import datetime
from google.colab import drive

# 1. SETUP: Critical Path Injections
PROJECT_ROOT = "/content/svomptr_9b"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def print_banner(text):
    print("\n" + "="*60)
    print(f"🚀 {datetime.now().strftime('%H:%M:%S')} | {text}")
    print("="*60 + "\n")

def keep_alive_heartbeat():
    """Background thread to keep the session active and show progress."""
    start_time = time.time()
    while True:
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"💓 [HEARTBEAT] Pipeline active | Elapsed: {elapsed} | Time: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(60) # Log every minute

def orchestrate():
    # Start Heartbeat
    heartbeat_thread = threading.Thread(target=keep_alive_heartbeat, daemon=True)
    heartbeat_thread.start()

    # Phase 0: Environment & Storage
    print_banner("PHASE 0: Initializing Google Drive Environment")
    try:
        if not os.path.exists('/content/drive'):
            print("🔗 Connecting to Google Drive...")
            drive.mount('/content/drive')
        else:
            print("✅ Google Drive already mounted.")
    except Exception as e:
        print(f"🛑 Drive Mount Failed: {e}")
        return

    # Define Persistent Storage (Brain)
    brain_dir = os.environ.get('SVOMPTR_BRAIN_PATH', '/content/drive/MyDrive/svomptr_brain')
    
    # Check/Create Folder Structure
    directories = ['datasets', 'checkpoints', 'weights', 'dop_alignment']
    for d in directories:
        path = os.path.join(brain_dir, d)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"📁 Created directory: {path}")
            
    dataset_file = os.path.join(brain_dir, 'datasets', 'synthetic_5M.jsonl')
    
    # Phase 1: High-Speed Generation via vLLM
    print_banner("PHASE 1: Dataset Generation (High-Speed vLLM)")
    
    # In Colab, we should prioritize check-pointing
    try:
        from svomptr.distillation.pipeline import run_distillation_pipeline
        
        print(f"🛠️ Starting Data Distillation Pipeline...")
        # use_vllm=True trigger high-quality synthetic data generation
        run_distillation_pipeline(
            output_file=dataset_file, 
            dry_run=False, 
            use_vllm=True
        )
        
    except Exception as e:
        print(f"🛑 Generation Phase Interrupted: {e}")
        print("💡 Suggestion: Check if GPU memory is full or Drive space is low.")
        return

    # Phase 2 & 3: Distillation & DOP Alignment
    print_banner("PHASE 2 & 3: Model Training & DOP Alignment")
    if os.path.exists(dataset_file):
        try:
            from svomptr_moe.train_chat_expert import train_chat_expert
            # This script handles both training and saving the DOP signature to Drive.
            train_chat_expert()
            print("✅ Model Training & DOP Alignment Complete.")
        except Exception as e:
            print(f"🛑 Training Phase Failed: {e}")
            return
    else:
        print("🛑 Error: Dataset file missing. Training cannot proceed.")

    print_banner("✨ SVOMPTR-9B FULL PIPELINE COMPLETED ✨")
    print(f"📦 Final Model Saved to: {os.path.join(brain_dir, 'weights', 'chat_expert_final')}")
    print(f"🎯 DOP Signature: {os.path.join(brain_dir, 'dop_alignment', 'alignment_signature.txt')}")

if __name__ == "__main__":
    # Ensure vllm is really installed (Colab-specific check)
    try:
        import vllm
    except ImportError:
        print_banner("INSTALLING SYSTEM DEPENDENCIES")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm", "tqdm", "transformers", "accelerate"])
        
    orchestrate()
