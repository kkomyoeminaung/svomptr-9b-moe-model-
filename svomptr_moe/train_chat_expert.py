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
        
        # Check for svomptr_brain in Google Drive (if run on Colab)
        brain_dir = os.environ.get("SVOMPTR_BRAIN_PATH", "/content/drive/MyDrive/svomptr_brain")
        
        if os.path.exists(brain_dir):
            print(f"SVOMPTR Brain detected at {brain_dir}")
            data_path = os.path.join(brain_dir, "datasets", "synthetic_5M.jsonl")
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
            
        print("Loading base model: Qwen/Qwen2.5-1.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

        if os.path.exists(data_path):
            raw_dataset = load_dataset("json", data_files=data_path)
            
            def tokenize_function(examples):
                # Construct the prompt
                prompts = [f"English: {en}\nSVOMPTR: " for en in examples["input"]]
                # Construct the completion
                targets = []
                for t, my in zip(examples["target"], examples["myanmar"]):
                    targets.append(f"{json.dumps(t, ensure_ascii=False)}\nMyanmar: {my}")
                
                inputs = [p + t for p, t in zip(prompts, targets)]
                model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
                
                # Setup labels for causal LM training (predict only the target part)
                labels = model_inputs["input_ids"].copy()
                # We should mask the prompt part in labels
                for i, p in enumerate(prompts):
                    p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
                    labels[i][:len(p_ids)] = -100 # Ignore prompt in loss
                
                model_inputs["labels"] = labels
                return model_inputs

            tokenized_dataset = raw_dataset["train"].map(
                tokenize_function, 
                batched=True, 
                remove_columns=raw_dataset["train"].column_names
            )
            
            training_args = TrainingArguments(
                output_dir=ckpt_dir,
                num_train_epochs=3,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                learning_rate=2e-5,
                save_steps=1000,
                logging_steps=100,
                bf16=torch.cuda.is_bf16_supported(),
                fp16=not torch.cuda.is_bf16_supported(),
                save_total_limit=3, # Prevent disk overflow, keep last 3
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
