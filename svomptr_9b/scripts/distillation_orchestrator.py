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

def keep_alive_heartbeat(stop_event):
    """Background thread to keep the session active and show progress."""
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"💓 [HEARTBEAT] Pipeline active | Elapsed: {elapsed} | Time: {datetime.now().strftime('%H:%M:%S')}")
        stop_event.wait(60) # Log every minute

def orchestrate():
    # Start Heartbeat
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(target=keep_alive_heartbeat, args=(stop_event,), daemon=True)
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
        stop_event.set()
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
    
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Phase 1: High-Speed Generation via vLLM
    print_banner("PHASE 1: Dataset Generation (High-Speed vLLM)")
    
    try:
        # We run the specific script to ensure standard behavior
        script_path = os.path.join(REPO_ROOT, "generate_data.py")
        print(f"🛠️ Executing Standalone Generator: python3 {script_path}")
        subprocess.run([sys.executable, script_path], check=True)
        print("✅ Data Generation Phase Finished.")
    except Exception as e:
        print(f"🛑 Generation Phase Interrupted: {e}")
        stop_event.set()
        return

    # Phase 2 & 3: Distillation & DOP Alignment
    print_banner("PHASE 2 & 3: Model Training & DOP Alignment")
    try:
        script_path = os.path.join(REPO_ROOT, "train_distillation.py")
        print(f"🛠️ Executing Standalone Trainer: python3 {script_path}")
        subprocess.run([sys.executable, script_path], check=True)
        print("✅ Model Training & DOP Alignment Finished.")
    except Exception as e:
        print(f"🛑 Training Phase Failed: {e}")
        stop_event.set()
        return

    print_banner("✨ SVOMPTR-9B FULL PIPELINE COMPLETED ✨")
    print(f"📦 Final Model Saved to: {os.path.join(brain_dir, 'weights', 'chat_expert_final')}")
    print(f"🎯 DOP Signature: {os.path.join(brain_dir, 'dop_alignment', 'alignment_signature.txt')}")
    
    stop_event.set()

if __name__ == "__main__":
    # Ensure vllm is really installed (Colab-specific check)
    try:
        import vllm
    except ImportError:
        print_banner("INSTALLING SYSTEM DEPENDENCIES")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm", "tqdm", "transformers", "accelerate"])
        
    orchestrate()
