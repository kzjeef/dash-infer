#!/usr/bin/env python3
"""Benchmark CUDA graph decode speedup."""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.runtime_config import AsModelRuntimeConfigBuilder

safe_model_name = 'qwen_Qwen2.5-7B-Instruct'
tmp_dir = '/tmp/dashinfer_test_output'

asgraph = os.path.join(tmp_dir, safe_model_name + '.asgraph')
if not os.path.exists(asgraph):
    print(f"ERROR: Serialized model not found at {tmp_dir}")
    sys.exit(1)

def run_test(label, enable_cuda_graph):
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
        .max_batch(1)
        .build())
    engine.install_model(runtime_cfg)
    engine.start_model(safe_model_name)

    from transformers import AutoTokenizer
    import torch
    model_path = os.path.expanduser('~/.cache/modelscope/hub/models/qwen/Qwen2___5-7B-Instruct')
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def infer(prompt, max_tokens):
        msgs = [{'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(text)
        t = torch.LongTensor(ids).contiguous()
        dl = torch.utils.dlpack.to_dlpack(t)
        gen_cfg = {'top_k': 1, 'max_length': max_tokens}
        status, handle, queue = engine.start_request(
            safe_model_name, {'input_ids': dl}, gen_cfg)
        if status != allspark.AsStatus.ALLSPARK_SUCCESS:
            return 0
        engine.sync_request(safe_model_name, handle)
        gen = queue.Get()
        n = len(gen.ids_from_generate)
        engine.release_request(safe_model_name, handle)
        return n

    # Warmup (uses engine internal warmup)
    # Then run actual test
    print(f'\n=== {label} ===')
    prompt = "Explain the theory of relativity in detail, covering both special and general relativity, their key equations, and experimental evidence."
    t0 = time.perf_counter()
    n = infer(prompt, max_tokens=128)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000
    if n > 0:
        print(f'{label}: {n} tokens in {total_ms:.1f}ms = {total_ms/n:.2f}ms/tok = {n/(total_ms/1000):.0f} tok/s')
    else:
        print(f'{label}: FAILED (0 tokens)')

    engine.stop_model(safe_model_name)
    engine.release_model(safe_model_name)
    return total_ms, n

# Run without CUDA graph
ms_eager, n_eager = run_test("Eager (no CUDA graph)", False)

# Run with CUDA graph
ms_graph, n_graph = run_test("CUDA Graph", True)

print(f'\n=== Comparison ===')
if n_eager > 0 and n_graph > 0:
    tpot_eager = ms_eager / n_eager
    tpot_graph = ms_graph / n_graph
    speedup = tpot_eager / tpot_graph if tpot_graph > 0 else 0
    print(f'Eager:      {tpot_eager:.2f} ms/tok ({n_eager/(ms_eager/1000):.0f} tok/s)')
    print(f'CUDA Graph: {tpot_graph:.2f} ms/tok ({n_graph/(ms_graph/1000):.0f} tok/s)')
    print(f'Speedup:    {speedup:.2f}x')
