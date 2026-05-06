# /generate_data.py
import os
import sys

# Critical Path Injection for Colab/Stand-alone run
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "svomptr_9b"))

from svomptr_9b.svomptr.distillation.pipeline import run_distillation_pipeline

def main():
    print("🚀 Starting Synthetic Data Generation Phase...")
    # Get brain path from environment or use default
    brain_dir = os.environ.get('SVOMPTR_BRAIN_PATH', '/content/drive/MyDrive/svomptr_brain')
    dataset_file = os.path.join(brain_dir, 'datasets', 'synthetic_5000000.jsonl')
    
    # Run the pipeline with 5M samples target
    # This will use vLLM if available and resume from checkpoints automatically
    run_distillation_pipeline(
        output_file=dataset_file,
        dry_run=False,
        use_vllm=True,
        target_total_samples=5000000
    )

if __name__ == "__main__":
    main()
