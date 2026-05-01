#!/bin/bash

echo "🚀 Starting SVOMPTR-9B MoE Pipeline"

# 1. Training
python3 -m svomptr_moe.train_chat_expert
python3 -m svomptr_moe.train_sub_experts
python3 -m svomptr_moe.train_router

# 2. Export
python3 -m svomptr_moe.export_gguf

# 3. Test
python3 -m svomptr_moe.inference_chat_first

echo "✅ Deployment Ready."
