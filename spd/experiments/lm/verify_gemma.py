#!/usr/bin/env python3
"""Verify Gemma-3-270M-IT model architecture and compatibility with SPD."""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def verify_gemma_model():
    """Load and analyze Gemma-3-270M-IT model structure."""

    print("=" * 80)
    print("Verifying Gemma-3-270M-IT Model Architecture")
    print("=" * 80)

    model_name = "google/gemma-3-270m-it"

    try:
        # Load tokenizer
        print(f"\n1. Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded successfully")
        print(f"  - Vocab size: {tokenizer.vocab_size}")
        print(f"  - Model max length: {tokenizer.model_max_length}")

        # Load model
        print(f"\n2. Loading model from {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16  # Just specify dtype, no device_map
        )
        print(f"✓ Model loaded successfully")
        print(f"  - Model class: {model.__class__.__name__}")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Analyze model structure
        print("\n3. Model Architecture Analysis:")
        print(f"  - Config: {model.config}")

        # Find MLP and attention modules
        print("\n4. Identifying target modules for SPD:")
        mlp_modules = []
        attn_modules = []
        all_modules = []

        for name, module in model.named_modules():
            all_modules.append(name)

            # Look for MLP projections
            if "mlp" in name and ("proj" in name or "gate" in name):
                mlp_modules.append(name)

            # Look for attention projections
            if ("attn" in name or "attention" in name) and "proj" in name:
                attn_modules.append(name)

        # Print first few layers to understand pattern
        print("\n  Sample layer structure (first 20 modules):")
        for i, name in enumerate(all_modules[:20]):
            print(f"    {name}")

        # Print identified patterns
        print("\n  MLP modules found:")
        for name in mlp_modules[:10]:  # Show first 10
            print(f"    {name}")
        if len(mlp_modules) > 10:
            print(f"    ... and {len(mlp_modules) - 10} more")

        print("\n  Attention modules found:")
        for name in attn_modules[:10]:  # Show first 10
            print(f"    {name}")
        if len(attn_modules) > 10:
            print(f"    ... and {len(attn_modules) - 10} more")

        # Extract pattern for config
        if mlp_modules:
            # Try to identify the pattern
            first_mlp = mlp_modules[0]
            print(f"\n5. Suggested target_module_patterns for SPD config:")

            # Determine the pattern structure
            if "gate_proj" in first_mlp:
                print('  - "model.layers.*.mlp.gate_proj"')
            if any("up_proj" in m for m in mlp_modules):
                print('  - "model.layers.*.mlp.up_proj"')
            if any("down_proj" in m for m in mlp_modules):
                print('  - "model.layers.*.mlp.down_proj"')

            # For attention
            if any("q_proj" in m for m in attn_modules):
                print('  - "model.layers.*.self_attn.q_proj"')
            if any("k_proj" in m for m in attn_modules):
                print('  - "model.layers.*.self_attn.k_proj"')
            if any("v_proj" in m for m in attn_modules):
                print('  - "model.layers.*.self_attn.v_proj"')
            if any("o_proj" in m for m in attn_modules):
                print('  - "model.layers.*.self_attn.o_proj"')

        # Test a simple forward pass
        print("\n6. Testing forward pass...")
        test_text = "Hello, this is a test."
        inputs = tokenizer(test_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {outputs.logits.shape}")

        print("\n" + "=" * 80)
        print("✅ Verification complete! Model is compatible with SPD.")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_gemma_model()
    sys.exit(0 if success else 1)