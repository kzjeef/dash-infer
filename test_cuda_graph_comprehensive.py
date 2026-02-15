#!/usr/bin/env python3
"""
Comprehensive CUDA Graph test suite.
Tests: execute modes, batch sizes, single-GPU, multi-GPU, correctness.

Usage:
  # Single GPU (default)
  CUDA_VISIBLE_DEVICES=0 python test_cuda_graph_comprehensive.py

  # 2-GPU
  CUDA_VISIBLE_DEVICES=0,1 python test_cuda_graph_comprehensive.py --gpus 0,1
"""
import os, sys, time, argparse, threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.prompt_utils import PromptTemplate


TESTS = [
    ("What is 2+3?", ["5"]),
    ("Capital of France in one word.", ["Paris"]),
    ("Chemical formula for water?", ["H2O"]),
]

def run_test(engine, model_loader, rt, model_name, prompt, max_out=64):
    tokenizer = model_loader.init_tokenizer().get_tokenizer()
    input_str = PromptTemplate.apply_chatml_template(prompt)
    input_len = len(tokenizer.encode(input_str))
    gen_cfg = model_loader.create_reference_generation_config_builder(rt)
    gen_cfg.update({'top_k': 1, 'max_length': input_len + max_out})
    status, handle, queue = engine.start_request_text(model_name, model_loader, input_str, gen_cfg)
    engine.sync_request(model_name, handle)
    elem = queue.Get()
    ids = list(elem.ids_from_generate)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    engine.release_request(model_name, handle)
    return text

def run_batch_test(engine, model_loader, rt, model_name, prompt, batch_size, max_out=32):
    """Fire batch_size concurrent requests, return all outputs."""
    tokenizer = model_loader.init_tokenizer().get_tokenizer()
    input_str = PromptTemplate.apply_chatml_template(prompt)
    input_len = len(tokenizer.encode(input_str))
    
    handles, queues = [], []
    for _ in range(batch_size):
        gen_cfg = model_loader.create_reference_generation_config_builder(rt)
        gen_cfg.update({'top_k': 1, 'max_length': input_len + max_out})
        status, handle, queue = engine.start_request_text(model_name, model_loader, input_str, gen_cfg)
        handles.append(handle)
        queues.append(queue)
    
    for h in handles:
        engine.sync_request(model_name, h)
    
    texts = []
    for h, q in zip(handles, queues):
        elem = q.Get()
        ids = list(elem.ids_from_generate)
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        engine.release_request(model_name, h)
    return texts


def setup_engine(model_path, model_name, device_ids, max_batch=16, serialize_dir=None):
    """Load model, serialize, create engine."""
    if serialize_dir is None:
        serialize_dir = f"/tmp/dashinfer_graph_test_{'_'.join(map(str, device_ids))}"
    
    model_loader = allspark.HuggingFaceModel(
        model_path, model_name,
        user_set_data_type='bfloat16',
        in_memory_serialize=False,
        trust_remote_code=True,
    )
    engine = allspark.Engine()
    (model_loader.load_model().read_model_config()
     .serialize_to_path(engine, serialize_dir, enable_quant=False,
                        weight_only_quant=True, skip_if_exists=True)
     .free_model())
    
    cfg = model_loader.create_reference_runtime_config_builder(
        model_name, TargetDevice.CUDA, device_ids, max_batch=max_batch)
    cfg.max_length(256)
    rt = cfg.build()
    engine.install_model(rt)
    engine.start_model(model_name)
    return engine, model_loader, rt


def test_correctness(engine, model_loader, rt, model_name, label=""):
    """Run basic correctness tests."""
    n_pass = 0
    for prompt, expects in TESTS:
        text = run_test(engine, model_loader, rt, model_name, prompt)
        ok = any(e.lower() in text.lower() for e in expects)
        n_pass += ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {prompt} → {text[:50]}")
    return n_pass, len(TESTS)


def test_batch_sizes(engine, model_loader, rt, model_name, sizes):
    """Test various batch sizes for correctness."""
    prompt, expects = TESTS[0]  # "What is 2+3?"
    n_pass = 0
    for bs in sizes:
        texts = run_batch_test(engine, model_loader, rt, model_name, prompt, bs)
        all_ok = all(any(e.lower() in t.lower() for e in expects) for t in texts)
        n_pass += all_ok
        bucket = 1
        while bucket < bs: bucket <<= 1
        print(f"    [{'PASS' if all_ok else 'FAIL'}] batch={bs:>2} bucket={bucket:>2} out[0]={texts[0][:30]}")
    return n_pass, len(sizes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0', help='Comma-separated GPU IDs')
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    args = parser.parse_args()
    
    device_ids = [int(x) for x in args.gpus.split(',')]
    n_gpus = len(device_ids)
    
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(args.model)
    model_name = "graph_test"
    
    total_pass, total_tests = 0, 0
    
    # Test 1: Eager mode correctness
    print(f"\n=== Test 1: Eager mode ({n_gpus}-GPU) ===")
    os.environ['ALLSPARK_EXECUTE_MODE'] = 'eager'
    engine, loader, rt = setup_engine(model_path, model_name, device_ids)
    p, t = test_correctness(engine, loader, rt, model_name)
    total_pass += p; total_tests += t
    engine.stop_model(model_name); engine.release_model(model_name)
    del engine
    os.environ.pop('ALLSPARK_EXECUTE_MODE', None)
    
    # Test 2: CudaGraph mode correctness
    print(f"\n=== Test 2: CudaGraph mode ({n_gpus}-GPU) ===")
    engine, loader, rt = setup_engine(model_path, model_name, device_ids)
    p, t = test_correctness(engine, loader, rt, model_name)
    total_pass += p; total_tests += t
    
    # Test 3: Batch sizes with graph
    print(f"\n=== Test 3: Various batch sizes ({n_gpus}-GPU) ===")
    batch_sizes = [1, 2, 3, 4, 5, 8]
    p, t = test_batch_sizes(engine, loader, rt, model_name, batch_sizes)
    total_pass += p; total_tests += t
    
    engine.stop_model(model_name); engine.release_model(model_name)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Total: {total_pass}/{total_tests} passed")
    if total_pass == total_tests:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    
    return 0 if total_pass == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
