import torch
import torch.nn as nn
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

def quantize_model(model: nn.Module, bits: int = 4):
    """
    Applies real quantization using bitsandbytes for efficient inference/training.
    """
    print(f"🔧 Applying Real {bits}-bit quantization to SVOMPTR-9B...")
    
    if bnb is None:
        print("⚠️ Warning: bitsandbytes not found. Falling back to FP16.")
        return model.half()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip lm_head or slot_predictor for stability if needed
            if any(skip in name for skip in ["lm_head", "slot_predictor"]):
                continue
                
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            
            if bits == 8:
                new_module = bnb.nn.Linear8bitLt(
                    in_features, out_features, bias=bias, has_fp16_weights=False
                )
            elif bits == 4:
                new_module = bnb.nn.Linear4bit(
                    in_features, out_features, bias=bias, bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                continue
                
            # Replace logic
            parent = model
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)
            
    print("✅ Quantization Applied Successfully.")
    return model
