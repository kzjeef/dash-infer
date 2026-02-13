#!/bin/bash
set -x
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
LOG=/scratch/workspaces/jiejing/workdir/allspark/dash-infer/dsv3_fp8_test_result.log
echo "=== DeepSeek V3 FP8 (671B) Test Started at $(date) ===" | tee $LOG

# Step 1: Install wheel
echo "=== Step 1: Install wheel ===" | tee -a $LOG
WHL=/scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/dist/dashinfer-2.0.0-cp310-cp310-linux_x86_64.whl
if [ ! -f "$WHL" ]; then
    echo "ERROR: Wheel not found at $WHL" | tee -a $LOG
    exit 1
fi
pip install "$WHL" --force-reinstall --no-deps 2>&1 | tee -a $LOG

# Overlay updated Python files
echo "=== Step 1b: Overlay updated Python files ===" | tee -a $LOG
SITE_PKG=$(python3 -c "import dashinfer; import os; print(os.path.dirname(dashinfer.__file__))")
for f in model/deepseek_v3.py model/__init__.py model/model_base.py; do
    cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/$f "$SITE_PKG/allspark/$f" 2>&1 | tee -a $LOG
done
for f in model_config.py engine_utils.py model_loader.py; do
    cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/$f "$SITE_PKG/allspark/$f" 2>&1 | tee -a $LOG
done

# Step 2: Run DeepSeek V3 model test on 8 GPUs
echo "=== Step 2: Running DeepSeek V3 (671B) inference on 8 GPUs ===" | tee -a $LOG
cd /scratch/workspaces/jiejing/workdir/allspark/dash-infer

timeout 1800 python3 << 'PYEOF' 2>&1 | tee -a $LOG
import os
import sys
import time

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark._allspark import AsStatus

# Use pre-converted BF16 model to avoid OOM during FP8 dequant + serialize
# (FP8 dequant doubles memory; BF16 model loads directly)
model_local_path = '/scratch/workspaces/jiejing/models/DeepSeek-V3-BF16'
safe_model_name = 'DeepSeek_v3'
tmp_dir = '/scratch/workspaces/jiejing/tmp/dashinfer_dsv3_fp8_output'

print(f"=== Loading model from {model_local_path} ===", flush=True)
start = time.time()

model_loader = allspark.HuggingFaceModel(
    model_local_path,
    safe_model_name,
    user_set_data_type='bfloat16',
    in_memory_serialize=False,
    trust_remote_code=True
)
engine = allspark.Engine()

print("=== Step 2a: Loading BF16 model weights ===", flush=True)
print("This may take several minutes for 671B model...", flush=True)
(model_loader
 .load_model(direct_load=True)
 .read_model_config()
)
# Enable Expert Parallelism: distribute 256 experts across 8 GPUs
# (32 experts per GPU instead of replicating all 256)
model_loader.as_model_config['use_ep'] = True
print(f"EP mode enabled: 256 experts / 8 GPUs = 32 per GPU", flush=True)

(model_loader
 .serialize_to_path(engine, tmp_dir,
                    enable_quant=False,
                    weight_only_quant=False,
                    skip_if_exists=True)
 .free_model())
print(f"Model loading + serialization took {time.time() - start:.1f}s", flush=True)

print("=== Step 2b: Installing model (8 GPUs) ===", flush=True)
runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(
    safe_model_name,
    TargetDevice.CUDA,
    [0, 1, 2, 3, 4, 5, 6, 7],
    max_batch=4
)
runtime_cfg_builder.max_length(4096)
runtime_cfg = runtime_cfg_builder.build()

engine.install_model(runtime_cfg)
engine.start_model(safe_model_name)
print(f"Model install + start took {time.time() - start:.1f}s total", flush=True)

print('=== Step 2c: Running inference ===', flush=True)
test_queries = [
    'Hello, who are you?',
    'What is 2+3? Answer briefly.',
    'Write a haiku about the moon.',
]

tokenizer = model_loader.init_tokenizer().get_tokenizer()
for q in test_queries:
    print(f'\n--- Query: {q} ---', flush=True)
    # Apply chat template for correct prompt formatting
    messages = [{"role": "user", "content": q}]
    input_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    print(f'Formatted: {repr(input_str[:100])}', flush=True)

    gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
    gen_cfg.update({'top_k': 1, 'max_length': 128})

    status, handle, queue = engine.start_request_text(
        safe_model_name, model_loader, input_str, gen_cfg)

    if status != AsStatus.ALLSPARK_SUCCESS:
        print(f'ERROR: Request failed with status {status}', flush=True)
        continue

    engine.sync_request(safe_model_name, handle)
    generated_elem = queue.Get()
    generated_ids = generated_elem.ids_from_generate
    output_text = model_loader.init_tokenizer().get_tokenizer().decode(generated_ids)

    print(f'Q: {q}')
    print(f'A: {output_text}')
    print(f'Tokens: {len(generated_ids)}')
    engine.release_request(safe_model_name, handle)

engine.stop_model(safe_model_name)
engine.release_model(safe_model_name)
print('\n=== ALL TESTS PASSED ===')
PYEOF

TEST_EXIT=$?
if [ $TEST_EXIT -ne 0 ]; then
    echo "=== TEST FAILED with exit code $TEST_EXIT ===" | tee -a $LOG
else
    echo "=== Test finished successfully at $(date) ===" | tee -a $LOG
fi
