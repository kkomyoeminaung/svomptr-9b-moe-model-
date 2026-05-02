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
            
    dataset_file = os.path.join(brain_dir, 'datasets', 'synthetic_5000000.jsonl')
    
    # Phase 1: High-Speed Generation via vLLM
    print_banner("PHASE 1: Dataset Generation (High-Speed vLLM)")
    
    try:
        # We run the specific script to ensure standard behavior
        print(f"🛠️ Executing Standalone Generator: python3 /generate_data.py")
        subprocess.run([sys.executable, "/generate_data.py"], check=True)
        print("✅ Data Generation Phase Finished.")
    except Exception as e:
        print(f"🛑 Generation Phase Interrupted: {e}")
        return

    # Phase 2 & 3: Distillation & DOP Alignment
    print_banner("PHASE 2 & 3: Model Training & DOP Alignment")
    try:
        print(f"🛠️ Executing Standalone Trainer: python3 /train_distillation.py")
        subprocess.run([sys.executable, "/train_distillation.py"], check=True)
        print("✅ Model Training & DOP Alignment Finished.")
    except Exception as e:
        print(f"🛑 Training Phase Failed: {e}")
        return

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
