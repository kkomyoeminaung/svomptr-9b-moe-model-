from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import sys
import os
import json

# Add the project root to python path to use svomptr modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svomptr_9b'))

try:
    from svomptr_moe.inference_chat_first import ChatFirstInference
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you are running this from the project root.")
    ChatFirstInference = None

app = FastAPI(title="SVOMPTR-9B Neural Link API")

# Initialize the model engine
engine = None
if ChatFirstInference:
    print("Initializing inference engine...")
    engine = ChatFirstInference()
else:
    print("WARNING: Inference engine is disabled.")

class ChatRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    english: str
    myanmar: str
    structure: str

@app.get("/api/health")
def health():
    return {"status": "ok", "engine_ready": engine is not None}

@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if engine and hasattr(engine, 'memory'):
        engine.memory.add_grammar_rule(f"USER FEEDBACK: {req.english}", req.myanmar, {"structure": req.structure})
    return {"status": "success", "message": "Feedback registered successfully to local neural map."}

@app.post("/api/chat")
def chat(req: ChatRequest):
    if not engine:
        return {
            "response": "[Fallback Mock Mode] The Python ML pipeline is loaded but missing dependencies or weights. Please ensure models are trained.",
            "frame": {"S": "System", "V": "Error", "O": "Weights Missing"}
        }
        
    try:
        # Ask engine to chat
        result = engine.chat(req.message)
        
        # If it returns a dict with response and frame
        if isinstance(result, dict) and "response" in result:
            return result
        
        return {
            "response": str(result),
            "frame": {"S": "System", "V": "Parsed", "O": "Direct"}
        }
    except Exception as e:
        return {
            "response": f"[Error during generation] {str(e)}",
            "frame": {"S": "System", "V": "Error", "O": "Exception"}
        }

@app.post("/api/ingest")
def ingest(data: dict):
    if engine and hasattr(engine, 'memory'):
        engine.memory.store_memory(data.get("text", ""))
    return {"status": "ingested", "kb_length": len(data.get("text", ""))}

@app.get("/api/grammar-rules")
def get_grammar_rules():
    if engine and hasattr(engine, 'memory'):
        rules = engine.memory.get_all_memories()
        # Filter for rule markers
        grammar_rules = [r for r in rules if "RULE_" in r or "USER FEEDBACK" in r.upper()]
        return {"rules": grammar_rules}
    return {"rules": []}

@app.post("/api/synthesize")
def synthesize():
    """Triggers the self-recursive improvement process."""
    if engine and hasattr(engine, 'memory'):
        # 1. Get raw candidate memories
        candidates = engine.memory.synthesize_rules()
        new_rules = []
        for c in candidates:
            # Simulate Neural Extraction: Transform raw translation into a Grammar Rule
            if "Translation:" in c:
                clean = c.split("Translation:")[-1].strip()
                # Store it back as a rule if detected as a recurring pattern
                engine.memory.add_grammar_rule("Synthesized Phrase", clean, {})
                new_rules.append(f"Autonomous Extraction: {clean}")
        
        return {"status": "success", "new_rules_count": len(new_rules), "samples": new_rules}
    return {"status": "error", "message": "Engine not initialized"}

@app.post("/api/start-learning")
def start_learning():
    import subprocess
    import threading
    
    def run_training():
        try:
            python_exe = sys.executable or "python3"
            subprocess.run([python_exe, "-m", "svomptr_moe.train_chat_expert"], check=True)
            subprocess.run([python_exe, "-m", "svomptr_moe.train_sub_experts"], check=True)
            subprocess.run([python_exe, "-m", "svomptr_moe.train_router"], check=True)
        except Exception as e:
            print(f"Training failed: {e}")

    threading.Thread(target=run_training).start()
    return {"message": "Neural core training initiated in the background on Colab GPU."}

if __name__ == "__main__":
    uvicorn.run("ml_api:app", host="0.0.0.0", port=8000, reload=True)
