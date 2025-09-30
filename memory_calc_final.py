import torch

# Given facts
num_params = 268_000_000  # 268M parameters
vocab_size = 262_144
seq_length = 256
batch_size_per_gpu = 3
hidden_size = 640
num_layers = 18
intermediate_size = 2048
num_components = 1000
num_target_modules = 54  # 3 MLP modules per layer × 18 layers
gate_hidden_dim = 12

bytes_per_float32 = 4
GB = 1024**3
MB = 1024**2

print("="*80)
print("FINAL ACCURATE MEMORY CALCULATION FOR GEMMA-270M SPD TRAINING")
print("="*80)

# 1. Base model memory
print("\n1. BASE MODEL MEMORY (frozen, not optimized):")
base_model_bytes = num_params * bytes_per_float32
base_model_gb = base_model_bytes / GB
print(f"   {num_params:,} params × 4 bytes = {base_model_gb:.2f} GB")

# 2. ComponentModel overhead
print("\n2. COMPONENTMODEL PARAMETERS:")

# Gate networks
gate_params_per_module = (
    hidden_size * gate_hidden_dim + gate_hidden_dim +  # input layer
    gate_hidden_dim * gate_hidden_dim + gate_hidden_dim +  # hidden layer
    gate_hidden_dim * num_components + num_components  # output layer
)
total_gate_params = gate_params_per_module * num_target_modules
gate_memory_gb = (total_gate_params * bytes_per_float32) / GB

# Component V and U matrices
# For each layer: 3 modules × (V[d_in×C] + U[C×d_out])
#   gate_proj: V[640×1000] + U[1000×2048]
#   up_proj: V[640×1000] + U[1000×2048]
#   down_proj: V[2048×1000] + U[1000×640]
component_params_per_layer = (
    2 * (hidden_size * num_components + num_components * intermediate_size) +  # gate_proj + up_proj
    (intermediate_size * num_components + num_components * hidden_size)  # down_proj
)
total_component_params = component_params_per_layer * num_layers
component_gb = (total_component_params * bytes_per_float32) / GB

total_trainable_params = total_gate_params + total_component_params
trainable_params_gb = gate_memory_gb + component_gb

print(f"   Gate networks: {total_gate_params:,} params = {gate_memory_gb:.2f} GB")
print(f"   Components (V+U): {total_component_params:,} params = {component_gb:.2f} GB")
print(f"   TOTAL TRAINABLE: {total_trainable_params:,} params = {trainable_params_gb:.2f} GB")

# 3. Optimizer state (Adam)
print("\n3. OPTIMIZER STATE (Adam: momentum + variance):")
optimizer_state_gb = 2 * trainable_params_gb
print(f"   2 × {trainable_params_gb:.2f} GB = {optimizer_state_gb:.2f} GB")

# 4. Gradients
print("\n4. GRADIENTS:")
gradient_gb = trainable_params_gb
print(f"   {gradient_gb:.2f} GB")

# 5. Single forward pass activations
print("\n5. SINGLE FORWARD PASS ACTIVATIONS:")

# Input embeddings
input_embed_bytes = batch_size_per_gpu * seq_length * hidden_size * bytes_per_float32
input_embed_mb = input_embed_bytes / MB

# Layer activations (simplified estimate)
# Each layer: attention (Q,K,V,out) + MLP (gate, up, down)
activations_per_layer_bytes = (
    4 * batch_size_per_gpu * seq_length * hidden_size +  # Attention
    2 * batch_size_per_gpu * seq_length * intermediate_size +  # MLP intermediate
    batch_size_per_gpu * seq_length * hidden_size  # MLP output
) * bytes_per_float32

total_layer_activations_gb = (num_layers * activations_per_layer_bytes) / GB

# Output logits (DOMINATES!)
output_logits_bytes = batch_size_per_gpu * seq_length * vocab_size * bytes_per_float32
output_logits_gb = output_logits_bytes / GB

single_forward_gb = (input_embed_bytes + num_layers * activations_per_layer_bytes + output_logits_bytes) / GB

print(f"   Input embeddings: {input_embed_mb:.2f} MB")
print(f"   Layer activations (18 layers): {total_layer_activations_gb:.2f} GB")
print(f"   Output logits [3x256x262144]: {output_logits_gb:.2f} GB << DOMINATES!")
print(f"   TOTAL SINGLE FORWARD: {single_forward_gb:.2f} GB")

# 6. CRITICAL: Layerwise reconstruction loss accumulates graphs
print("\n6. LAYERWISE RECONSTRUCTION LOSS (54 FORWARD PASSES):")
print("   Key insight: calc_masked_recon_layerwise_loss() does:")
print("   ")
print("   total_loss = 0")
print("   for module_name in mask_infos.items():  # 54 iterations")
print("       out = model.forward(batch, components={module_name})")
print("       loss = kl_divergence(out, target)")
print("       total_loss += loss  # <-- accumulates graph!")
print("   ")
print("   total_loss.backward()  # << NEEDS ALL 54 GRAPHS!")
print()
print("   Each forward pass creates:")
print(f"     - Full model activations: {single_forward_gb:.2f} GB")
print(f"     - Output logits: {output_logits_gb:.2f} GB")
print()
print("   However, NOT all activations are retained - only those needed for backprop.")
print("   Main memory cost: output logits (needed for KL loss gradient)")
print()

# Conservative estimate: PyTorch retains output logits + some intermediate activations
# for each of the 54 forward passes
retained_per_forward_gb = output_logits_gb + 0.1  # logits + small overhead
accumulated_graph_gb = num_target_modules * retained_per_forward_gb

print(f"   Retained per forward pass: ~{retained_per_forward_gb:.2f} GB")
print(f"   Accumulated over 54 passes: 54 × {retained_per_forward_gb:.2f} = {accumulated_graph_gb:.2f} GB")

# 7. Additional overhead
print("\n7. ADDITIONAL OVERHEAD:")
# Target model output (cached for all loss calculations)
cached_target_out_gb = output_logits_gb
# Causal importances (54 modules × batch × C)
ci_bytes = num_target_modules * batch_size_per_gpu * seq_length * num_components * bytes_per_float32
ci_gb = ci_bytes / GB
# PyTorch memory allocator overhead (~10-20% typical)
pytorch_overhead_gb = 2.0

print(f"   Cached target output: {cached_target_out_gb:.2f} GB")
print(f"   Causal importances: {ci_gb:.2f} GB")
print(f"   PyTorch allocator overhead: ~{pytorch_overhead_gb:.2f} GB")
misc_overhead_gb = cached_target_out_gb + ci_gb + pytorch_overhead_gb

# TOTAL MEMORY
print("\n" + "="*80)
print("TOTAL MEMORY BREAKDOWN:")
print("="*80)
print(f"1. Base model (frozen):     {base_model_gb:>8.2f} GB")
print(f"2. Trainable parameters:    {trainable_params_gb:>8.2f} GB")
print(f"3. Optimizer state:         {optimizer_state_gb:>8.2f} GB")
print(f"4. Gradients:               {gradient_gb:>8.2f} GB")
print(f"5. Accumulated graphs:      {accumulated_graph_gb:>8.2f} GB << MAIN ISSUE")
print(f"6. Misc overhead:           {misc_overhead_gb:>8.2f} GB")
print("-" * 80)

total_calculated_gb = (
    base_model_gb +
    trainable_params_gb +
    optimizer_state_gb +
    gradient_gb +
    accumulated_graph_gb +
    misc_overhead_gb
)
print(f"TOTAL CALCULATED:           {total_calculated_gb:>8.2f} GB")

observed_gb = 43.67
print(f"\nObserved PyTorch allocation: {observed_gb:>8.2f} GB")
diff_gb = total_calculated_gb - observed_gb
diff_pct = (diff_gb / observed_gb) * 100
print(f"Difference:                  {diff_gb:>8.2f} GB ({diff_pct:+.1f}%)")

# Analysis
print("\n" + "="*80)
print("MEMORY BREAKDOWN BY CATEGORY:")
print("="*80)
print(f"Model weights (base + components): {base_model_gb + trainable_params_gb:.2f} GB ({(base_model_gb + trainable_params_gb)/total_calculated_gb*100:.1f}%)")
print(f"Training state (opt + grads):      {optimizer_state_gb + gradient_gb:.2f} GB ({(optimizer_state_gb + gradient_gb)/total_calculated_gb*100:.1f}%)")
print(f"Computation graphs (54 passes):    {accumulated_graph_gb:.2f} GB ({accumulated_graph_gb/total_calculated_gb*100:.1f}%)")
print(f"Overhead:                          {misc_overhead_gb:.2f} GB ({misc_overhead_gb/total_calculated_gb*100:.1f}%)")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"1. Output logits: {output_logits_gb:.2f} GB per forward pass")
print(f"2. With 54 layerwise forward passes: {num_target_modules} × {output_logits_gb:.2f} GB = {num_target_modules * output_logits_gb:.2f} GB")
print(f"3. This represents {(num_target_modules * output_logits_gb)/total_calculated_gb*100:.1f}% of total memory!")
print(f"4. The bottleneck is vocab_size ({vocab_size:,}) causing massive logits")
print()
print("SOLUTION: The chunked KL divergence implementation is CRITICAL!")
print("Without it, memory would be completely exhausted.")