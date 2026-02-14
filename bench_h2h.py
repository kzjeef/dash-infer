#!/usr/bin/env python3
"""Head-to-head: SGLang (docker, GPU 7) vs DashInfer (GPU 0)."""
import os, sys, time, requests, json, concurrent.futures
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import AutoTokenizer

MODEL_PATH = os.path.expanduser('~/.cache/modelscope/hub/models/qwen/Qwen2___5-7B-Instruct')
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

SGLANG_URL = "http://localhost:30001"
BATCH_SIZES = [1, 8, 16]
MAX_TOKENS = 64
N_RUNS = 3  # average over multiple runs

PROMPTS = [
    "What is 2+3?", "Name the capital of France.",
    "Write a haiku about the ocean.", "Explain Newton's first law.",
    "What is the speed of light?", "Name three programming languages.",
    "What is the largest planet?", "Define entropy.",
    "What is photosynthesis?", "Who wrote Hamlet?",
    "What is the boiling point of water?", "Name a prime number greater than 10.",
    "What color is the sky?", "How many continents are there?",
    "What is DNA?", "Name the smallest country.",
]


def bench_sglang(batch_size):
    """Benchmark SGLang via HTTP API."""
    prompts = PROMPTS[:batch_size]
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]

    def do_req(msgs):
        r = requests.post(f"{SGLANG_URL}/v1/chat/completions",
            json={"model": "test", "messages": msgs,
                  "max_tokens": MAX_TOKENS, "temperature": 0}, timeout=60)
        return r.json()

    # Warmup
    do_req(messages_batch[0])

    best_tps = 0
    for run in range(N_RUNS):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
            responses = list(pool.map(do_req, messages_batch))
        t1 = time.perf_counter()

        total_tok = sum(r['usage']['completion_tokens'] for r in responses)
        wall_ms = (t1 - t0) * 1000
        tps = total_tok / (wall_ms / 1000)
        best_tps = max(best_tps, tps)

    return best_tps


def bench_dashinfer(batch_size, cuda_graph):
    """Benchmark DashInfer."""
    import torch, torch.utils.dlpack as dlpack
    from dashinfer import allspark
    from dashinfer.allspark.engine import TargetDevice
    from dashinfer.allspark.runtime_config import AsModelRuntimeConfigBuilder

    if cuda_graph:
        os.environ['ALLSPARK_CUDA_GRAPH'] = '1'
    else:
        os.environ.pop('ALLSPARK_CUDA_GRAPH', None)

    safe = 'qwen_Qwen2.5-7B-Instruct'
    engine = allspark.Engine()
    cfg = (AsModelRuntimeConfigBuilder()
        .model_name(safe)
        .model_dir('/tmp/dashinfer_test_output', safe)
        .compute_unit(TargetDevice.CUDA, [0], 0)
        .max_length(256).max_batch(max(BATCH_SIZES)).build())
    engine.install_model(cfg)
    engine.start_model(safe)

    prompts = PROMPTS[:batch_size]

    best_tps = 0
    for run in range(N_RUNS):
        handles, queues = [], []
        for p in prompts:
            msgs = [{'role': 'user', 'content': p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok.encode(text)
            t = torch.LongTensor([ids]).cpu().contiguous()
            dl = dlpack.to_dlpack(t)
            status, h, q = engine.start_request(safe, {'input_ids': dl},
                {'top_k': 1, 'max_length': MAX_TOKENS})
            if status == allspark.AsStatus.ALLSPARK_SUCCESS:
                handles.append(h); queues.append(q)

        t0 = time.perf_counter()
        for h in handles:
            engine.sync_request(safe, h)
        t1 = time.perf_counter()

        total_tok = sum(len(q.Get().ids_from_generate) for q in queues)
        for h in handles:
            engine.release_request(safe, h)

        wall_ms = (t1 - t0) * 1000
        tps = total_tok / (wall_ms / 1000) if wall_ms > 0 else 0
        best_tps = max(best_tps, tps)

    engine.stop_model(safe)
    engine.release_model(safe)
    return best_tps


print("=" * 70)
print("Head-to-head: Qwen2.5-7B-Instruct, bf16, max_tokens=64")
print("  DashInfer: GPU 0 (B200)  |  SGLang: GPU 7 (B200, docker)")
print(f"  Batch sizes: {BATCH_SIZES}  |  Best of {N_RUNS} runs")
print("=" * 70)

results = {}
for bs in BATCH_SIZES:
    print(f"\nBatch {bs}:")
    di_e = bench_dashinfer(bs, False)
    print(f"  DI eager:     {di_e:7.0f} tok/s")
    di_g = bench_dashinfer(bs, True)
    print(f"  DI+graph:     {di_g:7.0f} tok/s")
    sg = bench_sglang(bs)
    print(f"  SGLang:       {sg:7.0f} tok/s")
    results[bs] = (di_e, di_g, sg)

print("\n" + "=" * 70)
print(f"{'Batch':>5} | {'DI eager':>10} | {'DI+graph':>10} | {'SGLang':>10} | {'Graph/SG':>9}")
print("-" * 70)
for bs in BATCH_SIZES:
    di_e, di_g, sg = results[bs]
    ratio = di_g / sg if sg > 0 else 0
    print(f"{bs:>5} | {di_e:>9.0f} | {di_g:>9.0f} | {sg:>9.0f} | {ratio:>8.2f}x")
print("=" * 70)
