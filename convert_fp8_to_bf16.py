#!/usr/bin/env python3
"""Convert DeepSeek-V3 FP8 weights to BF16 shard-by-shard (memory-efficient).

Usage:
    python convert_fp8_to_bf16.py \
        --input /scratch/workspaces/jiejing/models/DeepSeek-V3 \
        --output /scratch/workspaces/jiejing/models/DeepSeek-V3-BF16
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequant_fp8_block(weight, scale_inv, block_size=(128, 128)):
    """Dequantize FP8 block-wise weight using scale_inv."""
    w_bf16 = weight.to(torch.bfloat16)
    M, N = w_bf16.shape
    bm, bn = block_size
    s = scale_inv.to(torch.bfloat16)
    s_expanded = s.repeat_interleave(bm, dim=0).repeat_interleave(bn, dim=1)
    s_expanded = s_expanded[:M, :N]
    return w_bf16 * s_expanded


def convert_shard(input_path, output_path, block_size=(128, 128)):
    """Convert a single safetensors shard from FP8 to BF16."""
    f = safe_open(input_path, framework='pt')
    keys = list(f.keys())

    # Group weight + scale pairs
    scale_keys = {k for k in keys if k.endswith('.weight_scale_inv')}
    weight_keys_with_scale = {k.replace('.weight_scale_inv', '.weight') for k in scale_keys}

    tensors = {}
    for key in keys:
        if key in scale_keys:
            # Skip scale tensors - they'll be consumed during dequant
            continue
        t = f.get_tensor(key)
        if key in weight_keys_with_scale:
            # This weight has an FP8 scale - dequantize
            scale_key = key.replace('.weight', '.weight_scale_inv')
            scale = f.get_tensor(scale_key)
            t = dequant_fp8_block(t, scale, block_size)
        elif t.dtype in (torch.float8_e4m3fn,):
            # FP8 without scale (shouldn't happen but handle it)
            t = t.to(torch.bfloat16)
        tensors[key] = t

    save_file(tensors, output_path)
    return len(tensors), len(scale_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input FP8 model directory')
    parser.add_argument('--output', required=True, help='Output BF16 model directory')
    parser.add_argument('--block-size', type=int, nargs=2, default=[128, 128])
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    block_size = tuple(args.block_size)

    # Copy non-safetensors files
    for f in input_dir.iterdir():
        if f.suffix != '.safetensors' and f.name != '.cache':
            dest = output_dir / f.name
            if f.is_file() and not dest.exists():
                shutil.copy2(f, dest)
                print(f"Copied {f.name}")
            elif f.is_dir() and not dest.exists():
                shutil.copytree(f, dest)
                print(f"Copied dir {f.name}")

    # Update config.json - remove quantization_config
    config_path = output_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config.pop('quantization_config', None)
        config['torch_dtype'] = 'bfloat16'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("Updated config.json (removed quantization_config)")

    # Convert safetensors shards
    shard_files = sorted(input_dir.glob('model-*.safetensors'))
    print(f"\nConverting {len(shard_files)} shards (block_size={block_size})...")

    total_start = time.time()
    for i, shard in enumerate(shard_files):
        out_shard = output_dir / shard.name
        if out_shard.exists():
            print(f"  [{i+1}/{len(shard_files)}] {shard.name} - already exists, skipping")
            continue

        start = time.time()
        n_tensors, n_scales = convert_shard(str(shard), str(out_shard), block_size)
        elapsed = time.time() - start
        print(f"  [{i+1}/{len(shard_files)}] {shard.name} -> {n_tensors} tensors, "
              f"{n_scales} dequantized, {elapsed:.1f}s")

    # Update index.json - remove scale entries
    index_path = output_dir / 'model.safetensors.index.json'
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        new_weight_map = {k: v for k, v in index['weight_map'].items()
                          if not k.endswith('.weight_scale_inv')}
        index['weight_map'] = new_weight_map
        # Recalculate total size (approximate: FP8 -> BF16 doubles size)
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"\nUpdated index: {len(new_weight_map)} entries (removed scale_inv)")

    total_elapsed = time.time() - total_start
    print(f"\nDone! Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
