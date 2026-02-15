#!/usr/bin/env python3
"""
Chunk Prefill: Accuracy Regression + Performance Benchmark
===========================================================
Tests correctness and performance of chunk prefill vs non-chunked prefill.

Usage:
  # Full test (single-batch + multi-batch, accuracy + perf):
  CUDA_VISIBLE_DEVICES=0 python test_chunk_prefill.py

  # Skip model conversion (reuse existing serialized model):
  CUDA_VISIBLE_DEVICES=0 python test_chunk_prefill.py --skip-convert

  # Only accuracy:
  CUDA_VISIBLE_DEVICES=0 python test_chunk_prefill.py --accuracy-only

  # Only perf:
  CUDA_VISIBLE_DEVICES=0 python test_chunk_prefill.py --perf-only
"""
import os, sys, time, argparse, json
from collections import defaultdict

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['ALLSPARK_TIME_LOG'] = '1'

import torch
import torch.utils.dlpack as dlpack
from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.runtime_config import AsModelRuntimeConfigBuilder
from transformers import AutoTokenizer

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = os.path.expanduser(
    '~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/'
    'snapshots/a09a35458c702b33eeacc393d103063234e8bc28')
SAFE_MODEL_NAME = 'qwen_Qwen2___5-7B-Instruct'
TMP_DIR = '/tmp/dashinfer_chunk_prefill_test'

# Chunk sizes to test: 0 = disabled (baseline)
CHUNK_SIZES = [0, 4096, 8192]

# Prompts for accuracy testing (greedy decode → deterministic)
ACCURACY_PROMPTS = [
    "What is 2+3? Answer with a single number.",
    "Name the capital of France in one word.",
    "What is the largest planet in our solar system?",
    "Define entropy in one sentence.",
    "Explain the Pythagorean theorem briefly.",
    "What is the chemical formula for water?",
    "Name three primary colors.",
    "Who wrote Romeo and Juliet?",
]

# Longer prompts for performance testing (to stress chunk prefill)
def make_long_prompt(base: str, target_tokens: int, tokenizer) -> str:
    """Pad a prompt to approximately target_tokens by repeating context."""
    padding = (" The quick brown fox jumps over the lazy dog." * 50 + "\n") * 10
    text = base + "\n\nContext:\n" + padding
    ids = tokenizer.encode(text)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=True)
    return text


# ─── Helpers ─────────────────────────────────────────────────────────────────

def convert_model(model_path: str, tmp_dir: str):
    """Convert HF model to DashInfer serialized format."""
    print(f"\n{'='*60}")
    print(f"Converting model: {model_path}")
    print(f"Output: {tmp_dir}")
    print(f"{'='*60}")

    engine = allspark.Engine()
    model_loader = allspark.HuggingFaceModel(
        model_path, SAFE_MODEL_NAME,
        user_set_data_type="bfloat16",
        in_memory_serialize=False,
        trust_remote_code=True)

    (model_loader.load_model()
     .read_model_config()
     .serialize_to_path(engine, tmp_dir, enable_quant=False,
                        weight_only_quant=False, skip_if_exists=True)
     .free_model())

    del engine
    print("Model conversion done.\n")
    return model_loader


def make_engine(max_length: int, max_batch: int, chunk_size: int,
                device_ids=None):
    """Create a DashInfer engine with given chunk prefill config."""
    if device_ids is None:
        device_ids = [0]

    engine = allspark.Engine()
    builder = (AsModelRuntimeConfigBuilder()
               .model_name(SAFE_MODEL_NAME)
               .model_dir(TMP_DIR, SAFE_MODEL_NAME)
               .compute_unit(TargetDevice.CUDA, device_ids, 0)
               .max_length(max_length)
               .max_batch(max_batch))

    if chunk_size > 0:
        builder.max_prefill_length(chunk_size)

    runtime_cfg = builder.build()
    engine.install_model(runtime_cfg)
    engine.start_model(SAFE_MODEL_NAME)
    return engine


def run_inference(engine, tokenizer, prompt: str, max_new_tokens: int = 64,
                  greedy: bool = True):
    """Run a single inference request and return generated token IDs."""
    msgs = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text)

    t = torch.LongTensor([input_ids]).cpu().contiguous()
    dl = dlpack.to_dlpack(t)

    gen_cfg = {
        'max_length': len(input_ids) + max_new_tokens,
        'top_k': 1 if greedy else 50,
        'top_p': 1.0 if greedy else 0.9,
        'temperature': 1.0,
    }

    status, handle, queue = engine.start_request(
        SAFE_MODEL_NAME, {'input_ids': dl}, gen_cfg)

    if status != allspark.AsStatus.ALLSPARK_SUCCESS:
        print(f"  ERROR: start_request failed: {status}")
        return [], 0.0

    t0 = time.perf_counter()
    engine.sync_request(SAFE_MODEL_NAME, handle)
    elapsed = time.perf_counter() - t0

    gen = queue.Get()
    gen_ids = list(gen.ids_from_generate) if gen else []

    engine.release_request(SAFE_MODEL_NAME, handle)
    return gen_ids, elapsed


def run_batch_inference(engine, tokenizer, prompts: list,
                        max_new_tokens: int = 64, greedy: bool = True):
    """Run a batch of concurrent inference requests."""
    handles = []
    queues = []
    input_lens = []

    t_start = time.perf_counter()

    for prompt in prompts:
        msgs = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(text)
        input_lens.append(len(input_ids))

        t = torch.LongTensor([input_ids]).cpu().contiguous()
        dl = dlpack.to_dlpack(t)

        gen_cfg = {
            'max_length': len(input_ids) + max_new_tokens,
            'top_k': 1 if greedy else 50,
            'top_p': 1.0 if greedy else 0.9,
            'temperature': 1.0,
        }

        status, handle, queue = engine.start_request(
            SAFE_MODEL_NAME, {'input_ids': dl}, gen_cfg)

        if status != allspark.AsStatus.ALLSPARK_SUCCESS:
            print(f"  ERROR: start_request failed: {status}")
            continue
        handles.append(handle)
        queues.append(queue)

    for handle in handles:
        engine.sync_request(SAFE_MODEL_NAME, handle)

    t_end = time.perf_counter()

    results = []
    for queue in queues:
        gen = queue.Get()
        gen_ids = list(gen.ids_from_generate) if gen else []
        results.append(gen_ids)

    for handle in handles:
        engine.release_request(SAFE_MODEL_NAME, handle)

    return results, t_end - t_start, input_lens


def teardown_engine(engine):
    """Clean up engine resources."""
    engine.stop_model(SAFE_MODEL_NAME)
    engine.release_model(SAFE_MODEL_NAME)
    del engine
    torch.cuda.empty_cache()
    time.sleep(1)


# ─── Accuracy Tests ──────────────────────────────────────────────────────────

def test_accuracy(tokenizer, max_length: int = 2048):
    """Compare greedy decode output across different chunk sizes."""
    print(f"\n{'='*70}")
    print(f"  ACCURACY REGRESSION TEST")
    print(f"  Chunk sizes: {CHUNK_SIZES}")
    print(f"  Prompts: {len(ACCURACY_PROMPTS)}")
    print(f"{'='*70}\n")

    # ── Single batch accuracy ──
    print("─── Single-Batch Accuracy ───")
    all_results = {}  # chunk_size -> {prompt_idx: gen_ids}

    for cs in CHUNK_SIZES:
        label = f"chunk={cs}" if cs > 0 else "baseline (no chunk)"
        print(f"\n  [{label}] Starting engine...")
        engine = make_engine(max_length, max_batch=1, chunk_size=cs)

        results = {}
        for i, prompt in enumerate(ACCURACY_PROMPTS):
            gen_ids, elapsed = run_inference(engine, tokenizer, prompt,
                                            max_new_tokens=64, greedy=True)
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            results[i] = gen_ids
            print(f"    [{i}] ({elapsed*1000:.0f}ms, {len(gen_ids)} tok) "
                  f"{prompt[:35]}... → {text[:50]}")

        all_results[cs] = results
        teardown_engine(engine)

    # Compare
    baseline = all_results[0]
    print(f"\n  ─── Single-Batch Comparison vs Baseline ───")
    single_pass = True
    for cs in CHUNK_SIZES:
        if cs == 0:
            continue
        label = f"chunk={cs}"
        match_count = 0
        for i in range(len(ACCURACY_PROMPTS)):
            if baseline.get(i) == all_results[cs].get(i):
                match_count += 1
            else:
                single_pass = False
                base_text = tokenizer.decode(baseline[i], skip_special_tokens=True)[:60]
                chunk_text = tokenizer.decode(all_results[cs][i], skip_special_tokens=True)[:60]
                print(f"    MISMATCH [{label}] prompt {i}: "
                      f"baseline='{base_text}' vs chunk='{chunk_text}'")
        status = "PASS" if match_count == len(ACCURACY_PROMPTS) else "FAIL"
        print(f"    {status}: {label} — {match_count}/{len(ACCURACY_PROMPTS)} match")

    # ── Multi-batch accuracy ──
    print(f"\n─── Multi-Batch Accuracy (batch={len(ACCURACY_PROMPTS)}) ───")
    batch_results = {}

    for cs in CHUNK_SIZES:
        label = f"chunk={cs}" if cs > 0 else "baseline (no chunk)"
        print(f"\n  [{label}] Starting engine...")
        engine = make_engine(max_length,
                            max_batch=len(ACCURACY_PROMPTS),
                            chunk_size=cs)

        results, elapsed, input_lens = run_batch_inference(
            engine, tokenizer, ACCURACY_PROMPTS,
            max_new_tokens=64, greedy=True)

        batch_results[cs] = results
        total_gen = sum(len(r) for r in results)
        print(f"    {len(results)} requests, {total_gen} gen tokens, "
              f"{elapsed*1000:.0f}ms")

        teardown_engine(engine)

    # Compare multi-batch
    baseline_batch = batch_results[0]
    print(f"\n  ─── Multi-Batch Comparison vs Baseline ───")
    multi_pass = True
    for cs in CHUNK_SIZES:
        if cs == 0:
            continue
        label = f"chunk={cs}"
        match_count = 0
        for i in range(len(ACCURACY_PROMPTS)):
            if i < len(baseline_batch) and i < len(batch_results[cs]):
                if baseline_batch[i] == batch_results[cs][i]:
                    match_count += 1
                else:
                    multi_pass = False
                    base_text = tokenizer.decode(baseline_batch[i], skip_special_tokens=True)[:60]
                    chunk_text = tokenizer.decode(batch_results[cs][i], skip_special_tokens=True)[:60]
                    print(f"    MISMATCH [{label}] prompt {i}: "
                          f"baseline='{base_text}' vs chunk='{chunk_text}'")
        status = "PASS" if match_count == len(ACCURACY_PROMPTS) else "FAIL"
        print(f"    {status}: {label} — {match_count}/{len(ACCURACY_PROMPTS)} match")

    overall = single_pass and multi_pass
    print(f"\n{'='*70}")
    print(f"  ACCURACY RESULT: {'ALL PASS ✓' if overall else 'FAILURES DETECTED ✗'}")
    print(f"{'='*70}")
    return overall


# ─── Performance Tests ───────────────────────────────────────────────────────

def test_performance(tokenizer, max_length: int = 16384):
    """Benchmark TTFT and throughput across chunk sizes and input lengths."""
    print(f"\n{'='*70}")
    print(f"  PERFORMANCE BENCHMARK")
    print(f"  Chunk sizes: {CHUNK_SIZES}")
    print(f"  Max length: {max_length}")
    print(f"{'='*70}\n")

    prefill_lengths = [512, 2048, 4096, 8192]
    prefill_lengths = [pl for pl in prefill_lengths if pl < max_length]

    base_prompt = "Summarize the following text in one sentence."

    # ── Single-batch TTFT benchmark ──
    print("─── Single-Batch TTFT Benchmark ───")
    print(f"{'Prefill Len':>12} {'Chunk Size':>12} {'TTFT (ms)':>12} "
          f"{'Gen Tokens':>12} {'Total (ms)':>12}")
    print("─" * 62)

    perf_results = []

    for cs in CHUNK_SIZES:
        engine = make_engine(max_length, max_batch=1, chunk_size=cs)

        for pl in prefill_lengths:
            long_prompt = make_long_prompt(base_prompt, pl, tokenizer)
            gen_ids, elapsed = run_inference(
                engine, tokenizer, long_prompt,
                max_new_tokens=32, greedy=True)

            label_cs = str(cs) if cs > 0 else "off"
            print(f"{pl:>12} {label_cs:>12} {elapsed*1000:>12.1f} "
                  f"{len(gen_ids):>12} {elapsed*1000:>12.1f}")

            perf_results.append({
                'mode': 'single',
                'prefill_len': pl,
                'chunk_size': cs,
                'ttft_ms': elapsed * 1000,
                'gen_tokens': len(gen_ids),
                'total_ms': elapsed * 1000,
            })

        teardown_engine(engine)

    # ── Multi-batch throughput benchmark ──
    batch_size = 4
    print(f"\n─── Multi-Batch Throughput (batch={batch_size}) ───")
    print(f"{'Prefill Len':>12} {'Chunk Size':>12} {'Wall (ms)':>12} "
          f"{'Total Tok':>12} {'Throughput':>12}")
    print("─" * 62)

    for cs in CHUNK_SIZES:
        engine = make_engine(max_length, max_batch=batch_size, chunk_size=cs)

        for pl in prefill_lengths:
            long_prompt = make_long_prompt(base_prompt, pl, tokenizer)
            prompts = [long_prompt] * batch_size

            results, elapsed, input_lens = run_batch_inference(
                engine, tokenizer, prompts,
                max_new_tokens=32, greedy=True)

            total_gen = sum(len(r) for r in results)
            tps = total_gen / elapsed if elapsed > 0 else 0
            label_cs = str(cs) if cs > 0 else "off"

            print(f"{pl:>12} {label_cs:>12} {elapsed*1000:>12.1f} "
                  f"{total_gen:>12} {tps:>12.0f} tok/s")

            perf_results.append({
                'mode': f'batch{batch_size}',
                'prefill_len': pl,
                'chunk_size': cs,
                'wall_ms': elapsed * 1000,
                'total_tokens': total_gen,
                'throughput_tps': tps,
            })

        teardown_engine(engine)

    # ── Summary ──
    print(f"\n─── Performance Summary ───")
    # Group by prefill_len and compare chunk vs no-chunk
    single_results = [r for r in perf_results if r['mode'] == 'single']
    for pl in prefill_lengths:
        baseline_r = [r for r in single_results
                      if r['prefill_len'] == pl and r['chunk_size'] == 0]
        if not baseline_r:
            continue
        base_ttft = baseline_r[0]['ttft_ms']
        print(f"  Prefill={pl}: baseline TTFT={base_ttft:.1f}ms", end="")
        for cs in CHUNK_SIZES:
            if cs == 0:
                continue
            chunk_r = [r for r in single_results
                       if r['prefill_len'] == pl and r['chunk_size'] == cs]
            if chunk_r:
                ratio = chunk_r[0]['ttft_ms'] / base_ttft if base_ttft > 0 else 0
                print(f"  | chunk={cs}: {chunk_r[0]['ttft_ms']:.1f}ms "
                      f"({ratio:.2f}x)", end="")
        print()

    return perf_results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Chunk Prefill Test')
    parser.add_argument('--skip-convert', action='store_true',
                        help='Skip model conversion (reuse existing)')
    parser.add_argument('--accuracy-only', action='store_true',
                        help='Only run accuracy tests')
    parser.add_argument('--perf-only', action='store_true',
                        help='Only run performance tests')
    parser.add_argument('--max-length', type=int, default=16384,
                        help='Engine max sequence length')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH,
                        help='Path to HuggingFace model')
    args = parser.parse_args()

    # Model setup
    global MODEL_PATH
    MODEL_PATH = args.model_path

    print(f"Model: {MODEL_PATH}")
    print(f"Serialized output: {TMP_DIR}")
    print(f"Max length: {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if not args.skip_convert:
        convert_model(MODEL_PATH, TMP_DIR)
    else:
        if not os.path.exists(os.path.join(TMP_DIR, SAFE_MODEL_NAME + '.asgraph')):
            print("No serialized model found — running conversion anyway")
            convert_model(MODEL_PATH, TMP_DIR)

    # Run tests
    accuracy_ok = True
    perf_results = []

    if not args.perf_only:
        accuracy_ok = test_accuracy(tokenizer, max_length=args.max_length)

    if not args.accuracy_only:
        perf_results = test_performance(tokenizer, max_length=args.max_length)

    # Save results
    results = {
        'accuracy_pass': accuracy_ok,
        'performance': perf_results,
        'config': {
            'model': MODEL_PATH,
            'max_length': args.max_length,
            'chunk_sizes': CHUNK_SIZES,
        }
    }
    results_path = os.path.join(TMP_DIR, 'chunk_prefill_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if not accuracy_ok:
        print("\n*** ACCURACY REGRESSION DETECTED ***")
        sys.exit(1)


if __name__ == '__main__':
    main()
