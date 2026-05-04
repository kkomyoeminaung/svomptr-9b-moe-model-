# SVOMPTR 9B MoE Model - Technical Specification

## 1. Overview
The SVOMPTR-9B model is a customized Large Language Model fine-tuned specifically for English-to-Myanmar translation. It uses the foundation of the **Qwen1.5-MoE-A2.7B** Sparse Mixture-of-Experts (SMoE) architecture. By employing MoE, the model efficiently manages computational cost while increasing overall model capacity to essentially 9 billion parameters, though only a subset (approximately 2.7 billion) are fully activated during inference.

## 2. Architecture & Parameters
- **Base Architecture**: Sparse Mixture of Experts (SMoE).
- **Base Model**: Qwen/Qwen1.5-MoE-A2.7B (a 9B parameter model routing to active 2.7B per token pass).
- **Total Parameters**: ~9 Billion.
- **Active Parameters (per token)**: ~2.7 Billion.
- **MoE Experts**: Typically 60 distinct experts with a Top-K routing protocol.
- **Context Length**: Trained with a context window limit of 2048 tokens. 

### Does it overfit or underfit?
- **Avoidance of Underfitting**: By selecting a 9B overall capacity MoE, the internal router delegates language generation and syntax formulation tasks across separate specialized Multi-Layer Perceptrons (MLPs). This vast parameter space gives it plenty of room to learn the complex English-Myanmar mapping. 
- **Avoidance of Overfitting**: The use of **QLoRA** (Quantized Low-Rank Adaptation) acts as robust built-in regularization. Because only adapters (a very small fraction of the 9B model—specifically `r=16`) are being trained, overfitting on the 100k dataset is heavily mitigated. Evaluation using validation splits (`5%` split assigned in the notebook) and the `cosine` learning rate schedule ensures smooth convergence rather than sharp memorization.

## 3. Training & Optimization Techniques (Anti-Bottleneck)
- **Quantization**: 4-bit NormalFloat (NF4) Quantization via `bitsandbytes` reduces memory footprint massively.
- **Adapters**: Applied to `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. Incorporating `gate_proj` and `down_proj` enables the model to uniquely customize its MoE behavior for translations.
- **Batch Size & Accumulation**: `per_device_train_batch_size=4` matched with `gradient_accumulation_steps=4` effectively mimics a larger batch size (16) without OOM crashing the T4 GPU.
- **vLLM Integration**: Included in the updated synthetic data pipeline to achieve near 4x/5x token-generation throughput compared to standard generation logic.

## 4. Dataset Flow & Security (No Leakage)
- **Train/Eval Split**: Training now actively utilizes `dataset.train_test_split(test_size=0.05, seed=42)`. Evaluating mid-training confirms generalizations without seeing test data, sealing the potential for Data Leakage.
- **Data Filtering (LaBSE)**: `Dataset_Analysis_and_Cleaning.ipynb` handles semantic mismatches. `SVOMPTR_9B_AutoTrain_Unsloth.ipynb` automatically prioritizes loading `synthetic_100k_cleaned.jsonl` if it exists.

## 5. Potential Bottlenecks for Production Scalability
If utilizing thousands of queries concurrently in production:
1. **Bottleneck**: Standard generation with HuggingFace pipeline in `Inference`.
   - **Solution**: The Notebook's Local API correctly leverages memory limits, but for enterprise usage, exporting the model to **GGUF** and running it via `vLLM` or `Ollama` natively gives you dynamic batching and Page Attention, removing KV Cache bottlenecks.
2. **Bottleneck**: Network transmission JSON parsing errors.
   - **Solution**: We implemented RegEx and explicit JSON fallbacks (`extract_json()`) within the pipeline to ensure that regardless of the model's chatty outputs, valid SVOMPTR pairs are successfully decoupled.
