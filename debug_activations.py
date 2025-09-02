#!/usr/bin/env python3
"""
Debug script to understand activation shapes from different layers in Gemma 2 7B
"""

import torch
from transformer_lens import HookedTransformer


def debug_activations():
    print("Loading Gemma 2 7B...")
    model = HookedTransformer.from_pretrained("gemma-7b", device="cpu")

    # Test prompt
    test_prompt = "The Eiffel Tower is in Paris.\n\nWhere is the Eiffel Tower located?\nA. Paris\nB. Berlin\nC. London\nD. Rome\n\nAnswer:"

    # Get all hook names
    all_hooks = list(model.hook_dict.keys())

    print(f"\nTotal number of hooks available: {len(all_hooks)}")
    print("\nFirst 30 hook names:")
    for i, hook in enumerate(all_hooks[:30]):
        print(f"  {i+1:2d}. {hook}")

    if len(all_hooks) > 30:
        print(f"  ... and {len(all_hooks) - 30} more")

    # Test a few key layers to see their shapes
    test_layers = [
        "blocks.0.hook_resid_post",
        "blocks.0.attn.hook_pattern",
        "blocks.0.mlp.hook_post",
        "blocks.0.ln1.hook_normalized",
        "blocks.0.ln2.hook_normalized",
        "ln_final.hook_normalized",
        "ln_final.hook_scale",
    ]

    activations = {}

    def save_activation(tensor, hook):
        activations[hook.name] = tensor.detach().clone()

    # Run the model with hooks
    with torch.no_grad():
        tokens = model.to_tokens(test_prompt)
        _ = model.run_with_hooks(
            tokens, fwd_hooks=[(layer, save_activation) for layer in test_layers]
        )

    print(f"\nInput tokens shape: {tokens.shape}")
    print(f"Number of tokens: {tokens.shape[1]}")

    for layer_name in test_layers:
        if layer_name in activations:
            tensor = activations[layer_name]
            print(f"\n{layer_name}:")
            print(f"  Full shape: {tensor.shape}")
            print(f"  Last token shape: {tensor[0, -1, :].shape}")
            print(f"  Flattened shape: {tensor[0, -1, :].flatten().shape}")

            # For attention patterns, show more details
            if "hook_pattern" in layer_name:
                print(f"  Attention heads: {tensor.shape[1]}")
                print(f"  Sequence length: {tensor.shape[2]}")
        else:
            print(f"\n{layer_name}: Not found")

    # Analyze layer types
    print(f"\n=== LAYER TYPE ANALYSIS ===")

    # Count different types of layers
    layer_types = {}
    other_layers = []
    for hook in all_hooks:
        if "hook_resid_post" in hook:
            layer_types["residual_post"] = layer_types.get("residual_post", 0) + 1
        elif "hook_pattern" in hook:
            layer_types["attention_pattern"] = (
                layer_types.get("attention_pattern", 0) + 1
            )
        elif "mlp.hook_post" in hook:
            layer_types["mlp_post"] = layer_types.get("mlp_post", 0) + 1
        elif "ln1.hook_normalized" in hook:
            layer_types["ln1_normalized"] = layer_types.get("ln1_normalized", 0) + 1
        elif "ln2.hook_normalized" in hook:
            layer_types["ln2_normalized"] = layer_types.get("ln2_normalized", 0) + 1
        elif "ln_final" in hook:
            layer_types["ln_final"] = layer_types.get("ln_final", 0) + 1
        elif "attn.hook_q" in hook:
            layer_types["attention_q"] = layer_types.get("attention_q", 0) + 1
        elif "attn.hook_k" in hook:
            layer_types["attention_k"] = layer_types.get("attention_k", 0) + 1
        elif "attn.hook_v" in hook:
            layer_types["attention_v"] = layer_types.get("attention_v", 0) + 1
        elif "attn.hook_result" in hook:
            layer_types["attention_result"] = layer_types.get("attention_result", 0) + 1
        elif "hook_embed" in hook:
            layer_types["embedding"] = layer_types.get("embedding", 0) + 1
        elif "hook_resid_pre" in hook:
            layer_types["residual_pre"] = layer_types.get("residual_pre", 0) + 1
        elif "hook_resid_mid" in hook:
            layer_types["residual_mid"] = layer_types.get("residual_mid", 0) + 1
        elif "hook_attn_in" in hook:
            layer_types["attention_input"] = layer_types.get("attention_input", 0) + 1
        elif "hook_mlp_in" in hook:
            layer_types["mlp_input"] = layer_types.get("mlp_input", 0) + 1
        elif "hook_attn_out" in hook:
            layer_types["attention_output"] = layer_types.get("attention_output", 0) + 1
        elif "hook_mlp_out" in hook:
            layer_types["mlp_output"] = layer_types.get("mlp_output", 0) + 1
        elif "mlp.hook_pre" in hook:
            layer_types["mlp_pre"] = layer_types.get("mlp_pre", 0) + 1
        elif "mlp.hook_pre_linear" in hook:
            layer_types["mlp_pre_linear"] = layer_types.get("mlp_pre_linear", 0) + 1
        elif "attn.hook_z" in hook:
            layer_types["attention_z"] = layer_types.get("attention_z", 0) + 1
        elif "attn.hook_attn_scores" in hook:
            layer_types["attention_scores"] = layer_types.get("attention_scores", 0) + 1
        elif "attn.hook_rot_k" in hook:
            layer_types["attention_rot_k"] = layer_types.get("attention_rot_k", 0) + 1
        elif "attn.hook_rot_q" in hook:
            layer_types["attention_rot_q"] = layer_types.get("attention_rot_q", 0) + 1
        elif "ln1.hook_scale" in hook:
            layer_types["ln1_scale"] = layer_types.get("ln1_scale", 0) + 1
        elif "ln2.hook_scale" in hook:
            layer_types["ln2_scale"] = layer_types.get("ln2_scale", 0) + 1
        else:
            layer_types["other"] = layer_types.get("other", 0) + 1
            other_layers.append(hook)

    print("Layer type counts:")
    for layer_type, count in sorted(layer_types.items()):
        print(f"  {layer_type}: {count}")

    print(f"\nSample of 'other' layers (first 20):")
    for i, layer in enumerate(other_layers[:20]):
        print(f"  {i+1:2d}. {layer}")

    if len(other_layers) > 20:
        print(f"  ... and {len(other_layers) - 20} more")

    # Show model architecture info
    print(f"\n=== MODEL ARCHITECTURE INFO ===")
    print(f"Number of layers: {model.cfg.n_layers}")
    print(f"Hidden dimension (d_model): {model.cfg.d_model}")
    print(f"Number of attention heads: {model.cfg.n_heads}")
    print(f"MLP expansion factor: {model.cfg.d_mlp // model.cfg.d_model}")
    print(f"MLP dimension: {model.cfg.d_mlp}")


if __name__ == "__main__":
    debug_activations()
