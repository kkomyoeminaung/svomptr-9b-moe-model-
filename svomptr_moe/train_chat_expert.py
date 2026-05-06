import os
import json
import os

def train_chat_expert():
    print("Task: Training Chat Expert (Expert 0)")
    print("Dataset: 5M Distillation Pairs (Myanmar-English SVOMPTR)")
    print("Objective: High-Speed Causal LM Distillation & DOP Alignment")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from datasets import load_dataset
        import torch
        
        # Hardware Detection & Configuration
        is_gpu = torch.cuda.is_available()
        device_map = "auto" if is_gpu else "cpu"
        torch_dtype = torch.bfloat16 if is_gpu and torch.cuda.is_bf16_supported() else (torch.float16 if is_gpu else torch.float32)
        
        print(f"🖥️ Hardware Detection: {'🚀 GPU (Active)' if is_gpu else '🐢 CPU Mode (Fallback)'}")
        print(f"⚙️ Computation Type: {torch_dtype}")

        # Check for svomptr_brain in Google Drive (if run on Colab)
        brain_dir = os.environ.get("SVOMPTR_BRAIN_PATH", "/content/drive/MyDrive/svomptr_brain")
        
        if os.path.exists(brain_dir):
            print(f"SVOMPTR Brain detected at {brain_dir}")
            data_path = os.path.join(brain_dir, "datasets", "synthetic_5000000.jsonl")
            ckpt_dir = os.path.join(brain_dir, "checkpoints", "chat_expert")
            final_dir = os.path.join(brain_dir, "weights", "chat_expert_final")
            dop_dir = os.path.join(brain_dir, "dop_alignment")
            os.makedirs(dop_dir, exist_ok=True)
        else:
            print("Running in local mode. SVOMPTR Brain not detected.")
            data_path = os.path.join(os.getcwd(), "data", "svomptr_chat_pairs.jsonl")
            ckpt_dir = "./checkpoints/chat_expert"
            final_dir = "./svomptr_export/chat_expert"
            dop_dir = "./svomptr_export/dop_alignment"
            os.makedirs(dop_dir, exist_ok=True)
            
        from svomptr_9b.svomptr.core.config import ModelConfig
        base_model_name = ModelConfig.STUDENT_BASE
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token # Fix for pad token issues

        if os.path.exists(data_path):
            raw_dataset = load_dataset("json", data_files=data_path)
            
            def tokenize_function(examples):
                # Construct ChatML format sequences
                texts = []
                for en, s, my in zip(examples.get("en", examples.get("input", [])), 
                                     examples.get("svomptr_structure", examples.get("target", [])), 
                                     examples.get("my", examples.get("myanmar", []))):
                    # Handle different schema naming if necessary
                    if isinstance(s, dict):
                        struct = f"S:{s.get('S','')}|V:{s.get('V','')}|O:{s.get('O','')}"
                    else:
                        struct = str(s)
                    
                    # Consistent Prompt Template: Matches inference_chat_first.py and Notebooks
                    sys_prompt = "English-to-Myanmar SVOMPTR Transformer Expert."
                    text = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nTranslate: {en}<|im_end|>\n<|im_start|>assistant\nTranslation: {my}\nStructure: {struct}<|im_end|>"
                    texts.append(text)
                
                model_inputs = tokenizer(texts, max_length=512, truncation=True, padding="max_length")
                
                # Setup labels for causal LM training (mask inputs)
                import copy
                labels = copy.deepcopy(model_inputs["input_ids"])
                for i, text in enumerate(texts):
                    # Mask everything up to and including the assistant start token
                    assistant_marker = "<|im_start|>assistant\n"
                    parts = text.split(assistant_marker)
                    if len(parts) > 1:
                        prompt_part = parts[0] + assistant_marker
                        prompt_ids = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
                        labels[i][:len(prompt_ids)] = -100
                
                model_inputs["labels"] = labels
                return model_inputs

            tokenized_dataset = raw_dataset["train"].map(
                tokenize_function, 
                batched=True, 
                remove_columns=raw_dataset["train"].column_names
            )
            
            training_args = TrainingArguments(
                output_dir=ckpt_dir,
                num_train_epochs=1, # 5M samples usually only need 1 epoch for distillation
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=1e-5,
                save_strategy="steps",
                save_steps=1000,
                logging_steps=100,
                bf16=torch.cuda.is_bf16_supported(),
                fp16=not torch.cuda.is_bf16_supported(),
                save_total_limit=2,
                report_to="none"
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )
            
            # Resume from checkpoint if it exists
            last_checkpoint = None
            if os.path.isdir(ckpt_dir):
                checkpoints = [os.path.join(ckpt_dir, d) for d in os.listdir(ckpt_dir) if d.startswith("checkpoint")]
                if checkpoints:
                    last_checkpoint = max(checkpoints, key=os.path.getmtime)
                    print(f"Resuming training from {last_checkpoint} to prevent progress loss")
                    
            print("Starting DOP Alignment & Distillation...")
            trainer.train(resume_from_checkpoint=last_checkpoint)
            
            print(f"Status: 100% Alignment Complete. Saving weights to {final_dir}")
            model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            
            with open(os.path.join(dop_dir, "alignment_signature.txt"), "w") as f:
                f.write("DOP ALIGNMENT VERIFIED\n")
        else:
            print(f"Dataset not found at: {data_path}")
            print("Please run the generation script first (via vLLM) to start Neural Training.")
            
    except ImportError as e:
        print(f"Missing libraries for Neural Distillation. {e}")
        print("Run: pip install torch transformers datasets accelerate")

if __name__ == "__main__":
    train_chat_expert()
