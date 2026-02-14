#!/usr/bin/env python3
"""Benchmark CUDA graph at various batch sizes."""
import os, sys, time
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

MAX_BATCH = 16
MAX_TOKENS = 64

# Pool of prompts to draw from
PROMPT_POOL = [
    "What is 2+3?",
    "Name the capital of France.",
    "Write a haiku about the ocean.",
    "Explain Newton's first law.",
    "What is the speed of light?",
    "Name three programming languages.",
    "What is the largest planet?",
    "Define entropy.",
    "What is photosynthesis?",
    "Who wrote Hamlet?",
    "What is the boiling point of water?",
    "Name a prime number greater than 10.",
    "What color is the sky?",
    "How many continents are there?",
    "What is DNA?",
    "Name the smallest country.",
]


def make_engine(enable_cuda_graph):
    if enable_cuda_graph:
        os.environ['ALLSPARK_CUDA_GRAPH'] = '1'
    else:
        os.environ.pop('ALLSPARK_CUDA_GRAPH', None)

    engine = allspark.Engine()
    runtime_cfg = (AsModelRuntimeConfigBuilder()
        .model_name(safe_model_name)
        .model_dir(tmp_dir, safe_model_name)
        .compute_unit(TargetDevice.CUDA, [0], 0)
        .max_length(256)
        .max_batch(MAX_BATCH)
        .build())
    engine.install_model(runtime_cfg)
    engine.start_model(safe_model_name)
    return engine


def run_batch_test(engine, batch_size):
    """Send batch_size requests, wait for all, return results."""
    prompts = PROMPT_POOL[:batch_size]
    handles = []
    queues = []

    for prompt in prompts:
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
            print(f"    Request failed: {status}")
            return None
        handles.append(handle)
        queues.append(queue)

    t_start = time.perf_counter()
    for handle in handles:
        engine.sync_request(safe_model_name, handle)
    t_end = time.perf_counter()

    total_tokens = 0
    results = []
    for i, queue in enumerate(queues):
        gen = queue.Get()
        ids = list(gen.ids_from_generate)
        total_tokens += len(ids)
        text = tok.decode(ids, skip_special_tokens=True)
        results.append((ids, text))

    for handle in handles:
        engine.release_request(safe_model_name, handle)

    wall_ms = (t_end - t_start) * 1000
    tps = total_tokens / (wall_ms / 1000) if wall_ms > 0 else 0
    return {
        'batch': batch_size,
        'tokens': total_tokens,
        'wall_ms': wall_ms,
        'tps': tps,
        'results': results,
    }


BATCH_SIZES = [1, 7, 8, 13, 15, 16]

print("=" * 60)
print(f"Benchmarking batch sizes: {BATCH_SIZES}")
print(f"Max tokens per request: {MAX_TOKENS}")
print("=" * 60)

# Run eager
print("\n--- Eager Mode ---")
engine_eager = make_engine(False)
eager_data = {}
for bs in BATCH_SIZES:
    r = run_batch_test(engine_eager, bs)
    if r:
        eager_data[bs] = r
        print(f"  batch={bs:2d}: {r['tokens']:3d} tok in {r['wall_ms']:7.1f}ms = {r['tps']:7.0f} tok/s")
engine_eager.stop_model(safe_model_name)
engine_eager.release_model(safe_model_name)
del engine_eager

# Run CUDA graph
print("\n--- CUDA Graph Mode ---")
engine_graph = make_engine(True)
graph_data = {}
for bs in BATCH_SIZES:
    r = run_batch_test(engine_graph, bs)
    if r:
        graph_data[bs] = r
        print(f"  batch={bs:2d}: {r['tokens']:3d} tok in {r['wall_ms']:7.1f}ms = {r['tps']:7.0f} tok/s")
engine_graph.stop_model(safe_model_name)
engine_graph.release_model(safe_model_name)
del engine_graph

# Correctness check
print("\n--- Correctness ---")
all_ok = True
for bs in BATCH_SIZES:
    if bs not in eager_data or bs not in graph_data:
        print(f"  batch={bs}: SKIP (missing data)")
        continue
    e_results = eager_data[bs]['results']
    g_results = graph_data[bs]['results']
    match = all(e[0] == g[0] for e, g in zip(e_results, g_results))
    print(f"  batch={bs:2d}: {'PASS' if match else 'FAIL'}")
    if not match:
        all_ok = False
        for i, (e, g) in enumerate(zip(e_results, g_results)):
            if e[0] != g[0]:
                print(f"    req {i}: eager='{e[1][:40]}' graph='{g[1][:40]}'")
                break

# Summary table
print("\n" + "=" * 60)
print(f"{'Batch':>5} | {'Eager tok/s':>12} | {'Graph tok/s':>12} | {'Speedup':>8}")
print("-" * 60)
for bs in BATCH_SIZES:
    if bs in eager_data and bs in graph_data:
        e_tps = eager_data[bs]['tps']
        g_tps = graph_data[bs]['tps']
        speedup = g_tps / e_tps if e_tps > 0 else 0
        print(f"{bs:>5} | {e_tps:>12.0f} | {g_tps:>12.0f} | {speedup:>7.2f}x")
print("=" * 60)
print(f"Correctness: {'ALL PASS' if all_ok else 'SOME FAIL'}")
