#!/usr/bin/env python3
"""Calculate expected memory usage for Gemma-3-270m SPD training."""

# Gemma-3-270m architecture (from verify_gemma.py output)
HIDDEN_SIZE = 640
INTERMEDIATE_SIZE = 2048
NUM_LAYERS = 18
VOCAB_SIZE = 262144
TOTAL_MODEL_PARAMS = 268_098_176

# SPD configuration (from gemma_270m_multilingual_config.yaml)
C = 1000
GATE_HIDDEN_DIMS = [12]  # Corrected from config - was 10.7x too large
NUM_MODULES = 54  # 18 layers × 3 MLP projections

# Training configuration
MICROBATCH_SIZE = 3  # batch_size=9 / 3 GPUs = 3 per GPU
SEQ_LEN = 256
GRADIENT_ACCUM_STEPS = 3
BYTES_PER_PARAM = 4  # fp32

def format_gb(bytes_val):
    return f"{bytes_val / 1e9:.3f} GB"

print("=" * 80)
print("MEMORY CALCULATION FOR GEMMA-3-270M SPD TRAINING")
print("=" * 80)
print(f"\nArchitecture: {NUM_LAYERS} layers, hidden={HIDDEN_SIZE}, intermediate={INTERMEDIATE_SIZE}")
print(f"SPD Config: C={C}, {NUM_MODULES} modules, microbatch={MICROBATCH_SIZE}")
print("\n" + "-" * 80)

# 1. Target Model (frozen)
target_model_mem = TOTAL_MODEL_PARAMS * BYTES_PER_PARAM
print(f"\n1. TARGET MODEL (frozen):")
print(f"   {TOTAL_MODEL_PARAMS:,} params × {BYTES_PER_PARAM} bytes = {format_gb(target_model_mem)}")

# 2. Component Parameters (U, V matrices)
# For each MLP projection: gate_proj, up_proj, down_proj
# gate_proj: [intermediate, hidden] → U: [intermediate, C], V: [C, hidden]
# up_proj: [intermediate, hidden] → U: [intermediate, C], V: [C, hidden]
# down_proj: [hidden, intermediate] → U: [hidden, C], V: [C, intermediate]

gate_proj_params = INTERMEDIATE_SIZE * C + C * HIDDEN_SIZE
up_proj_params = INTERMEDIATE_SIZE * C + C * HIDDEN_SIZE
down_proj_params = HIDDEN_SIZE * C + C * INTERMEDIATE_SIZE

comp_params_per_layer = gate_proj_params + up_proj_params + down_proj_params
total_comp_params = comp_params_per_layer * NUM_LAYERS
comp_mem = total_comp_params * BYTES_PER_PARAM

print(f"\n2. COMPONENT PARAMETERS (U, V):")
print(f"   gate_proj per layer: {gate_proj_params:,} params")
print(f"   up_proj per layer: {up_proj_params:,} params")
print(f"   down_proj per layer: {down_proj_params:,} params")
print(f"   Total per layer: {comp_params_per_layer:,} params")
print(f"   Total all layers: {total_comp_params:,} params × {BYTES_PER_PARAM} bytes = {format_gb(comp_mem)}")

# 3. Gate Parameters (vector MLP: hidden_size → 128 → 1, per component)
# For each component: gate takes hidden_size input, 128 hidden, 1 output
# Total params per component: hidden_size * 128 + 128 * 1
gate_params_per_comp = HIDDEN_SIZE * GATE_HIDDEN_DIMS[0] + GATE_HIDDEN_DIMS[0] * 1
gate_params_per_module = C * gate_params_per_comp
total_gate_params = NUM_MODULES * gate_params_per_module
gate_mem = total_gate_params * BYTES_PER_PARAM

print(f"\n3. GATE PARAMETERS (vector MLP):")
print(f"   Params per component: {gate_params_per_comp:,}")
print(f"   Params per module: {gate_params_per_module:,}")
print(f"   Total: {total_gate_params:,} params × {BYTES_PER_PARAM} bytes = {format_gb(gate_mem)}")

# 4. Optimizer State (AdamW: exp_avg + exp_avg_sq)
trainable_params = total_comp_params + total_gate_params
optimizer_state_mem = trainable_params * 2 * BYTES_PER_PARAM

print(f"\n4. OPTIMIZER STATE (AdamW: exp_avg + exp_avg_sq):")
print(f"   {trainable_params:,} trainable params × 2 × {BYTES_PER_PARAM} bytes = {format_gb(optimizer_state_mem)}")

# 5. Gradients
gradient_mem = trainable_params * BYTES_PER_PARAM

print(f"\n5. GRADIENTS:")
print(f"   {trainable_params:,} params × {BYTES_PER_PARAM} bytes = {format_gb(gradient_mem)}")

# 6. Activations (still in scope at optimizer.step())
# After my cleanup, these SHOULD be deleted, but let's check what was there before

target_out_mem = MICROBATCH_SIZE * SEQ_LEN * VOCAB_SIZE * BYTES_PER_PARAM
pre_weight_acts_mem = NUM_MODULES * MICROBATCH_SIZE * SEQ_LEN * HIDDEN_SIZE * BYTES_PER_PARAM
causal_importances_mem = NUM_MODULES * MICROBATCH_SIZE * SEQ_LEN * C * BYTES_PER_PARAM
causal_importances_upper_leaky_mem = causal_importances_mem  # Same shape
weight_deltas_mem = total_comp_params * BYTES_PER_PARAM  # Same as component params
batch_mem = MICROBATCH_SIZE * SEQ_LEN * BYTES_PER_PARAM  # Token IDs (int32)

print(f"\n6. ACTIVATIONS (should be deleted by cleanup):")
print(f"   target_out: [{MICROBATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE}] = {format_gb(target_out_mem)}")
print(f"   pre_weight_acts: {NUM_MODULES} × [{MICROBATCH_SIZE}, {SEQ_LEN}, {HIDDEN_SIZE}] = {format_gb(pre_weight_acts_mem)}")
print(f"   causal_importances: {NUM_MODULES} × [{MICROBATCH_SIZE}, {SEQ_LEN}, {C}] = {format_gb(causal_importances_mem)}")
print(f"   causal_importances_upper_leaky: {NUM_MODULES} × [{MICROBATCH_SIZE}, {SEQ_LEN}, {C}] = {format_gb(causal_importances_upper_leaky_mem)}")
print(f"   weight_deltas: {total_comp_params:,} params = {format_gb(weight_deltas_mem)}")
print(f"   batch: [{MICROBATCH_SIZE}, {SEQ_LEN}] = {format_gb(batch_mem)}")
total_activations = (target_out_mem + pre_weight_acts_mem + causal_importances_mem +
                     causal_importances_upper_leaky_mem + weight_deltas_mem + batch_mem)
print(f"   TOTAL ACTIVATIONS: {format_gb(total_activations)}")

# 7. Base memory (without activations)
base_mem = target_model_mem + comp_mem + gate_mem + optimizer_state_mem + gradient_mem
print(f"\n7. BASE MEMORY (model + optimizer + gradients, NO activations):")
print(f"   {format_gb(base_mem)}")

# 8. Total WITH activations (before cleanup)
total_with_activations = base_mem + total_activations
print(f"\n8. TOTAL WITH ACTIVATIONS (before cleanup):")
print(f"   {format_gb(total_with_activations)}")

# 9. Expected after cleanup
expected_after_cleanup = base_mem
print(f"\n9. EXPECTED AFTER CLEANUP (activations deleted):")
print(f"   {format_gb(expected_after_cleanup)}")

print(f"\n10. ACTUAL OBSERVED:")
print(f"   PyTorch allocated: 19.22 GB")
print(f"   Free memory: 0.076 GB")
print(f"   Total used: 19.30 GB (approx)")

print(f"\n11. DISCREPANCY ANALYSIS:")
expected_gb = expected_after_cleanup / 1e9
actual_gb = 19.22
discrepancy_gb = actual_gb - expected_gb
print(f"   Expected after cleanup: {expected_gb:.3f} GB")
print(f"   Actual observed: {actual_gb:.3f} GB")
print(f"   DISCREPANCY: {discrepancy_gb:.3f} GB")

if discrepancy_gb > 0.5:
    print(f"\n   ⚠️  MEMORY LEAK DETECTED: {discrepancy_gb:.3f} GB unaccounted for!")
    print(f"   Possible sources:")
    print(f"   - Computation graphs not freed (retain_graph=True)")
    print(f"   - Hidden references to activations")
    print(f"   - PyTorch memory fragmentation")
    print(f"   - DDP buffers/hooks")
else:
    print(f"\n   ✓ Memory usage within expected range")

print("\n" + "=" * 80)
