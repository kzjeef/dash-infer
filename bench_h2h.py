#!/usr/bin/env python3
"""Head-to-head: SGLang (docker) vs DashInfer eager vs DashInfer+CUDAGraph.
Generates 300+ tokens per request."""
import os, sys, time, requests, json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch, torch.utils.dlpack as dlpack
from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.runtime_config import AsModelRuntimeConfigBuilder
from transformers import AutoTokenizer

MODEL_PATH = os.path.expanduser(
    '~/.cache/modelscope/hub/models/qwen/Qwen2___5-7B-Instruct')
SGLANG_URL = "http://localhost:30001"
SGLANG_MODEL = "/models/qwen/Qwen2___5-7B-Instruct"
DI_MODEL = 'qwen_Qwen2.5-7B-Instruct'
DI_DIR = '/tmp/dashinfer_test_output'
MAX_TOKENS = 384
N_RUNS = 3

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

PROMPT = ("Write a detailed explanation of how neural networks learn through "
          "backpropagation, covering the chain rule, gradient computation, "
          "weight updates, learning rate, and common optimizers like Adam and SGD.")

def bench_sglang():
    print(f"\n{'='*60}")
    print(f"  SGLang (docker, CUDA graph, batch=1)")
    print(f"{'='*60}")
    requests.post(f"{SGLANG_URL}/v1/chat/completions",
        json={"model": SGLANG_MODEL,
              "messages": [{"role": "user", "content": "Hello"}],
              "max_tokens": 16, "temperature": 0}, timeout=30)
    times, gen_counts = [], []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        r = requests.post(f"{SGLANG_URL}/v1/chat/completions",
            json={"model": SGLANG_MODEL,
                  "messages": [{"role": "user", "content": PROMPT}],
                  "max_tokens": MAX_TOKENS, "temperature": 0}, timeout=120)
        t1 = time.perf_counter()
        resp = r.json()
        n = resp['usage']['completion_tokens']
        elapsed = t1 - t0
        times.append(elapsed); gen_counts.append(n)
        print(f"  Run {i+1}: {n} tok in {elapsed*1000:.0f} ms  ({n/elapsed:.0f} tok/s)")
    avg = sum(times)/len(times); avg_n = sum(gen_counts)/len(gen_counts)
    print(f"  Avg: {avg*1000:.0f} ms, {avg_n/avg:.0f} tok/s")
    return avg_n / avg

def bench_dashinfer(enable_cuda_graph):
    label = "DashInfer+CUDAGraph" if enable_cuda_graph else "DashInfer (eager)"
    print(f"\n{'='*60}")
    print(f"  {label} (batch=1)")
    print(f"{'='*60}")
    if enable_cuda_graph:
        os.environ['ALLSPARK_CUDA_GRAPH'] = '1'
    else:
        os.environ.pop('ALLSPARK_CUDA_GRAPH', None)
    engine = allspark.Engine()
    runtime_cfg = (AsModelRuntimeConfigBuilder()
        .model_name(DI_MODEL).model_dir(DI_DIR, DI_MODEL)
        .compute_unit(TargetDevice.CUDA, [0], 0)
        .max_length(512).max_batch(1).build())
    engine.install_model(runtime_cfg)
    engine.start_model(DI_MODEL)
    def infer(prompt, max_tok):
        msgs = [{'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(text)
        t = torch.LongTensor([ids]).cpu().contiguous()
        dl = dlpack.to_dlpack(t)
        status, handle, queue = engine.start_request(
            DI_MODEL, {'input_ids': dl}, {'top_k': 1, 'max_length': max_tok})
        if status != allspark.AsStatus.ALLSPARK_SUCCESS:
            return 0, 0.0
        t0 = time.perf_counter()
        engine.sync_request(DI_MODEL, handle)
        t1 = time.perf_counter()
        gen = queue.Get(); n = len(gen.ids_from_generate)
        engine.release_request(DI_MODEL, handle)
        return n, t1 - t0
    infer("Hello", 64); infer("Count to ten", 128)
    times, gen_counts = [], []
    for i in range(N_RUNS):
        n, elapsed = infer(PROMPT, MAX_TOKENS)
        times.append(elapsed); gen_counts.append(n)
        print(f"  Run {i+1}: {n} tok in {elapsed*1000:.0f} ms  ({n/elapsed:.0f} tok/s)")
    avg = sum(times)/len(times); avg_n = sum(gen_counts)/len(gen_counts)
    print(f"  Avg: {avg*1000:.0f} ms, {avg_n/avg:.0f} tok/s")
    engine.stop_model(DI_MODEL); engine.release_model(DI_MODEL); del engine
    return avg_n / avg

print(f"\nQwen2.5-7B-Instruct, 1xB200, bf16, batch=1, ~{MAX_TOKENS} gen tokens")
di_eager_tps = bench_dashinfer(False)
di_graph_tps = bench_dashinfer(True)
sg_tps = bench_sglang()

print(f"\n{'='*60}")
print(f"  Summary (tok/s, higher is better)")
print(f"{'='*60}")
print(f"  DashInfer eager:       {di_eager_tps:6.0f} tok/s")
print(f"  DashInfer+CUDAGraph:   {di_graph_tps:6.0f} tok/s  ({di_graph_tps/di_eager_tps:.0%} of eager)")
print(f"  SGLang (docker):       {sg_tps:6.0f} tok/s")
print(f"  DI+Graph / SGLang:     {di_graph_tps/sg_tps:.2f}x")
print(f"{'='*60}")
