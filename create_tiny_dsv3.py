#!/usr/bin/env python3
"""Create a tiny (2-layer) DeepSeek-V3 model with random BF16 weights for testing.

This creates a model with the same architecture as DeepSeek-V3 but with:
- 2 layers instead of 61
- 8 experts instead of 256
- Same MLA dimensions (kv_lora_rank=512, q_lora_rank=1536, etc.)

Usage:
    python create_tiny_dsv3.py --output /scratch/workspaces/jiejing/models/tiny-dsv3
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def create_tiny_dsv3(output_dir, num_layers=2, num_experts=8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model config matching DeepSeek-V3 architecture
    hidden_size = 7168
    intermediate_size = 18432
    moe_intermediate_size = 2048
    num_attention_heads = 128
    kv_lora_rank = 512
    q_lora_rank = 1536
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    vocab_size = 1024  # Small vocab for testing
    first_k_dense_replace = 1  # 1 dense layer, rest MoE
    n_shared_experts = 1

    config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_deepseek.DeepseekV3Config",
            "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
        },
        "model_type": "deepseek_v3",
        "torch_dtype": "bfloat16",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": moe_intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "vocab_size": vocab_size,
        "n_routed_experts": num_experts,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": first_k_dense_replace,
        "n_shared_experts": n_shared_experts,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "rope_scaling": {
            "beta_fast": 32, "beta_slow": 1, "factor": 40,
            "mscale": 1.0, "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096, "type": "yarn"
        },
        "max_position_embeddings": 4096,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "num_nextn_predict_layers": 0,
        "ep_size": 1,
        "moe_layer_freq": 1,
    }

    # Output hidden size for attention
    # o_proj: (hidden_size, num_heads * v_head_dim) = (7168, 128*128) = (7168, 16384)
    # q_b_proj: (num_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank)
    #         = (128 * 192, 1536) = (24576, 1536)
    # kv_b_proj: (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
    #          = (128 * 256, 512) = (32768, 512)
    # kv_a_proj_with_mqa: (kv_lora_rank + qk_rope_head_dim, hidden_size) = (576, 7168)

    def rand_bf16(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16) * 0.01

    tensors = {}
    weight_map = {}

    # Global weights
    tensors["model.embed_tokens.weight"] = rand_bf16(vocab_size, hidden_size)
    tensors["model.norm.weight"] = torch.ones(hidden_size, dtype=torch.bfloat16)
    tensors["lm_head.weight"] = rand_bf16(vocab_size, hidden_size)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # Layer norms
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size, dtype=torch.bfloat16)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size, dtype=torch.bfloat16)

        # MLA attention weights
        tensors[f"{prefix}.self_attn.q_a_proj.weight"] = rand_bf16(q_lora_rank, hidden_size)
        tensors[f"{prefix}.self_attn.q_a_layernorm.weight"] = torch.ones(q_lora_rank, dtype=torch.bfloat16)
        tensors[f"{prefix}.self_attn.q_b_proj.weight"] = rand_bf16(
            num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank)
        tensors[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = rand_bf16(
            kv_lora_rank + qk_rope_head_dim, hidden_size)
        tensors[f"{prefix}.self_attn.kv_a_layernorm.weight"] = torch.ones(kv_lora_rank, dtype=torch.bfloat16)
        tensors[f"{prefix}.self_attn.kv_b_proj.weight"] = rand_bf16(
            num_attention_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = rand_bf16(
            hidden_size, num_attention_heads * v_head_dim)

        if i < first_k_dense_replace:
            # Dense FFN
            tensors[f"{prefix}.mlp.gate_proj.weight"] = rand_bf16(intermediate_size, hidden_size)
            tensors[f"{prefix}.mlp.up_proj.weight"] = rand_bf16(intermediate_size, hidden_size)
            tensors[f"{prefix}.mlp.down_proj.weight"] = rand_bf16(hidden_size, intermediate_size)
        else:
            # MoE layer
            tensors[f"{prefix}.mlp.gate.weight"] = rand_bf16(num_experts, hidden_size)
            tensors[f"{prefix}.mlp.gate.e_score_correction_bias"] = torch.zeros(
                num_experts, dtype=torch.bfloat16)

            # Shared expert
            tensors[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = rand_bf16(
                moe_intermediate_size, hidden_size)
            tensors[f"{prefix}.mlp.shared_experts.up_proj.weight"] = rand_bf16(
                moe_intermediate_size, hidden_size)
            tensors[f"{prefix}.mlp.shared_experts.down_proj.weight"] = rand_bf16(
                hidden_size, moe_intermediate_size)

            # Routed experts
            for j in range(num_experts):
                tensors[f"{prefix}.mlp.experts.{j}.gate_proj.weight"] = rand_bf16(
                    moe_intermediate_size, hidden_size)
                tensors[f"{prefix}.mlp.experts.{j}.up_proj.weight"] = rand_bf16(
                    moe_intermediate_size, hidden_size)
                tensors[f"{prefix}.mlp.experts.{j}.down_proj.weight"] = rand_bf16(
                    hidden_size, moe_intermediate_size)

    # Save as single safetensors file
    shard_name = "model-00001-of-000001.safetensors"
    safetensors_path = output_dir / shard_name
    save_file(tensors, str(safetensors_path))

    # Create index
    for key in tensors:
        weight_map[key] = shard_name

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy configuration_deepseek.py from the real model
    src_config_py = Path("/scratch/workspaces/jiejing/models/DeepSeek-V3/configuration_deepseek.py")
    if src_config_py.exists():
        import shutil
        shutil.copy2(src_config_py, output_dir / "configuration_deepseek.py")

    # Create a simple tokenizer config (use the real one if available)
    src_tokenizer = Path("/scratch/workspaces/jiejing/models/DeepSeek-V3")
    for tok_file in ["tokenizer.json", "tokenizer_config.json"]:
        src = src_tokenizer / tok_file
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / tok_file)

    n_params = sum(t.numel() for t in tensors.values())
    print(f"Created tiny DeepSeek-V3 model:")
    print(f"  Layers: {num_layers}")
    print(f"  Experts: {num_experts}")
    print(f"  Dense layers: {first_k_dense_replace}")
    print(f"  Parameters: {n_params / 1e6:.1f}M")
    print(f"  Size on disk: {total_size / 1e9:.2f}GB")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-experts', type=int, default=8)
    args = parser.parse_args()
    create_tiny_dsv3(args.output, args.num_layers, args.num_experts)
