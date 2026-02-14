#!/usr/bin/env python3
"""Verify CUDA graph decode produces identical output to eager mode."""
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

if not os.path.exists(os.path.join(tmp_dir, safe_model_name + '.asgraph')):
    print(f"ERROR: Serialized model not found at {tmp_dir}")
    sys.exit(1)

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


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
        .max_batch(1)
        .build())
    engine.install_model(runtime_cfg)
    engine.start_model(safe_model_name)
    return engine


def infer(engine, prompt, max_tokens=64):
    msgs = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok.encode(text)

    # 2D tensor [1, seq_len] as expected by _start_request
    t = torch.LongTensor([ids]).cpu().contiguous()
    dl = dlpack.to_dlpack(t)
    gen_cfg = {'top_k': 1, 'max_length': max_tokens}

    status, handle, queue = engine.start_request(
        safe_model_name, {'input_ids': dl}, gen_cfg)
    if status != allspark.AsStatus.ALLSPARK_SUCCESS:
        print(f"  start_request failed: {status}")
        return [], ""
    engine.sync_request(safe_model_name, handle)
    gen = queue.Get()
    out_ids = list(gen.ids_from_generate)
    engine.release_request(safe_model_name, handle)
    out_text = tok.decode(out_ids, skip_special_tokens=True)
    return out_ids, out_text


prompts = [
    "What is 2+3?",
    "Name the capital of France.",
    "Write a haiku about the ocean.",
    "Explain Newton's first law in one sentence.",
]

print("=== Testing Eager Mode ===")
engine_eager = make_engine(False)
eager_results = []
for p in prompts:
    ids, text = infer(engine_eager, p, max_tokens=64)
    eager_results.append((ids, text))
    print(f"  [{len(ids)} tok] {p} -> {text[:80]}")
engine_eager.stop_model(safe_model_name)
engine_eager.release_model(safe_model_name)
del engine_eager

print("\n=== Testing CUDA Graph Mode ===")
engine_graph = make_engine(True)
graph_results = []
for p in prompts:
    ids, text = infer(engine_graph, p, max_tokens=64)
    graph_results.append((ids, text))
    print(f"  [{len(ids)} tok] {p} -> {text[:80]}")
engine_graph.stop_model(safe_model_name)
engine_graph.release_model(safe_model_name)
del engine_graph

print("\n=== Correctness Comparison ===")
all_match = True
for i, p in enumerate(prompts):
    eager_ids, eager_text = eager_results[i]
    graph_ids, graph_text = graph_results[i]
    match = eager_ids == graph_ids
    status = "PASS" if match else "FAIL"
    if not match:
        all_match = False
    print(f"  {status}: \"{p}\"")
    if not match:
        print(f"    Eager ({len(eager_ids)} tok): {eager_text[:100]}")
        print(f"    Graph ({len(graph_ids)} tok): {graph_text[:100]}")
        # Show first divergence
        for j in range(min(len(eager_ids), len(graph_ids))):
            if eager_ids[j] != graph_ids[j]:
                print(f"    First diff at token {j}: eager={eager_ids[j]} graph={graph_ids[j]}")
                break

print(f"\n{'ALL TESTS PASSED' if all_match else 'SOME TESTS FAILED'}")
sys.exit(0 if all_match else 1)
