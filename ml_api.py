from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import sys
import os
import json

# Add the project root to python path to use svomptr modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

@app.get("/api/health")
def health():
    return {"status": "ok", "engine_ready": engine is not None}

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
    return {"status": "ingested", "kb_length": len(data.get("text", ""))}

@app.post("/api/start-learning")
def start_learning():
    import subprocess
    import threading
    
    def run_training():
        try:
            subprocess.run(["python", "-m", "svomptr_moe.train_chat_expert"], check=True)
            subprocess.run(["python", "-m", "svomptr_moe.train_sub_experts"], check=True)
            subprocess.run(["python", "-m", "svomptr_moe.train_router"], check=True)
        except Exception as e:
            print(f"Training failed: {e}")

    threading.Thread(target=run_training).start()
    return {"message": "Neural core training initiated in the background on Colab GPU."}

if __name__ == "__main__":
    uvicorn.run("ml_api:app", host="0.0.0.0", port=8000, reload=True)
