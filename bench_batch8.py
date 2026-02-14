#!/usr/bin/env python3
"""Benchmark CUDA graph with batch=8."""
import os, sys, time, threading
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.utils.dlpack as dlpack
from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.runtime_config import AsModelRuntimeConfigBuilder
from transformers import AutoTokenizer

safe_model_name = 'qwen_Qwen2.5-7B-Instruct'
tmp_dir = '/tmp/dashinfer_test_output'
model_path = os.path.expanduser(
    '~/.cache/modelscope/hub/models/qwen/Qwen2___5-7B-Instruct')
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

BATCH = 8
MAX_TOKENS = 64

prompts = [
    "What is 2+3?",
    "Name the capital of France.",
    "Write a haiku about the ocean.",
    "Explain Newton's first law in one sentence.",
    "What is the speed of light?",
    "Name three programming languages.",
    "What is the largest planet in our solar system?",
    "Define entropy in one sentence.",
]

def make_engine(enable_cuda_graph):
    if enable_cuda_graph:
        os.environ['ALLSPARK_CUDA_GRAPH'] = '1'
    else:
        os.environ.pop('ALLSPARK_CUDA_GRAPH', None)
    os.environ['ALLSPARK_TIME_LOG'] = '1'

    engine = allspark.Engine()
    runtime_cfg = (AsModelRuntimeConfigBuilder()
        .model_name(safe_model_name)
        .model_dir(tmp_dir, safe_model_name)
        .compute_unit(TargetDevice.CUDA, [0], 0)
        .max_length(256)
        .max_batch(BATCH)
        .build())
    engine.install_model(runtime_cfg)
    engine.start_model(safe_model_name)
    return engine


def run_batch(engine, label):
    """Send BATCH requests concurrently and wait for all to complete."""
    handles = []
    queues = []

    t_start = time.perf_counter()

    # Start all requests
    for i, prompt in enumerate(prompts[:BATCH]):
        msgs = [{'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok.encode(text)
        t = torch.LongTensor([ids]).cpu().contiguous()
        dl = dlpack.to_dlpack(t)
        gen_cfg = {'top_k': 1, 'max_length': MAX_TOKENS}
        status, handle, queue = engine.start_request(
            safe_model_name, {'input_ids': dl}, gen_cfg)
        if status != allspark.AsStatus.ALLSPARK_SUCCESS:
            print(f"  Request {i} failed: {status}")
            continue
        handles.append(handle)
        queues.append((i, queue))

    # Wait for all
    for handle in handles:
        engine.sync_request(safe_model_name, handle)
    t_end = time.perf_counter()

    total_tokens = 0
    results = {}
    for i, queue in queues:
        gen = queue.Get()
        ids = list(gen.ids_from_generate)
        total_tokens += len(ids)
        text = tok.decode(ids, skip_special_tokens=True)
        results[i] = (ids, text)

    # Release all
    for handle in handles:
        engine.release_request(safe_model_name, handle)

    total_ms = (t_end - t_start) * 1000
    print(f"\n{label}:")
    print(f"  Batch size: {len(handles)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Wall time: {total_ms:.1f}ms")
    print(f"  Throughput: {total_tokens / (total_ms / 1000):.0f} tok/s")
    if total_tokens > 0:
        print(f"  Avg TPOT: {total_ms / (total_tokens / len(handles)):.2f}ms")
    for i in sorted(results.keys()):
        ids, text = results[i]
        print(f"  [{len(ids)} tok] {prompts[i][:30]}... -> {text[:50]}")

    return total_ms, total_tokens, results


print("=== Eager (batch=8) ===")
engine = make_engine(False)
ms_e, tok_e, res_e = run_batch(engine, "Eager")
engine.stop_model(safe_model_name)
engine.release_model(safe_model_name)
del engine

print("\n=== CUDA Graph (batch=8) ===")
engine = make_engine(True)
ms_g, tok_g, res_g = run_batch(engine, "CUDA Graph")
engine.stop_model(safe_model_name)
engine.release_model(safe_model_name)
del engine

# Correctness check
print("\n=== Correctness ===")
all_match = True
for i in sorted(res_e.keys()):
    if i not in res_g:
        print(f"  MISSING: prompt {i}")
        all_match = False
        continue
    match = res_e[i][0] == res_g[i][0]
    print(f"  {'PASS' if match else 'FAIL'}: {prompts[i][:40]}")
    if not match:
        all_match = False
        print(f"    Eager: {res_e[i][1][:60]}")
        print(f"    Graph: {res_g[i][1][:60]}")

print(f"\n{'ALL MATCH' if all_match else 'MISMATCH'}")

# Summary
print(f"\n=== Summary (batch={BATCH}) ===")
if tok_e > 0 and tok_g > 0:
    tps_e = tok_e / (ms_e / 1000)
    tps_g = tok_g / (ms_g / 1000)
    print(f"  Eager:      {tps_e:.0f} tok/s")
    print(f"  CUDA Graph: {tps_g:.0f} tok/s")
    print(f"  Speedup:    {tps_g / tps_e:.2f}x")
