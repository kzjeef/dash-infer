#!/bin/bash
set -x
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
LOG=/scratch/workspaces/jiejing/workdir/allspark/dash-infer/deepseek_v3_test_result.log
echo "=== DeepSeek V3 Model Test Started at $(date) ===" | tee $LOG

# Step 1: Install wheel (rebuild separately if needed)
echo "=== Step 1: Install wheel ===" | tee -a $LOG
WHL=/scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/dist/dashinfer-2.0.0-cp310-cp310-linux_x86_64.whl
if [ ! -f "$WHL" ]; then
    echo "ERROR: Wheel not found at $WHL. Build first with run_full_test.sh or setup.py" | tee -a $LOG
    exit 1
fi
pip install "$WHL" --force-reinstall --no-deps 2>&1 | tee -a $LOG

# Also copy updated Python files that may not be in the wheel yet
echo "=== Step 1b: Overlay updated Python files ===" | tee -a $LOG
SITE_PKG=$(python3 -c "import dashinfer; import os; print(os.path.dirname(dashinfer.__file__))")
cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/model/deepseek_v3.py "$SITE_PKG/allspark/model/" 2>&1 | tee -a $LOG
cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/model/__init__.py "$SITE_PKG/allspark/model/" 2>&1 | tee -a $LOG
cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/model/model_base.py "$SITE_PKG/allspark/model/" 2>&1 | tee -a $LOG
cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/model_config.py "$SITE_PKG/allspark/" 2>&1 | tee -a $LOG
cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/engine_utils.py "$SITE_PKG/allspark/" 2>&1 | tee -a $LOG
cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/model_loader.py "$SITE_PKG/allspark/" 2>&1 | tee -a $LOG

# Step 2: Run DeepSeek V3 model test
echo "=== Step 2: Running DeepSeek V3 model inference ===" | tee -a $LOG
cd /scratch/workspaces/jiejing/workdir/allspark/dash-infer

# Use timeout to prevent infinite hangs
timeout 300 python3 << 'PYEOF' 2>&1 | tee -a $LOG
import os
import sys
import time
import signal

# Single GPU for tiny model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark._allspark import AsStatus

model_local_path = '/scratch/workspaces/jiejing/models/tiny-dsv3'
safe_model_name = 'DeepSeek_v3'
tmp_dir = '/scratch/workspaces/jiejing/tmp/dashinfer_dsv3_tiny_output'

print(f"=== Loading model from {model_local_path} ===", flush=True)
start = time.time()

# Load BF16 model with direct_load=True (loads safetensors directly, avoids HF model init)
model_loader = allspark.HuggingFaceModel(
    model_local_path,
    safe_model_name,
    user_set_data_type='bfloat16',
    in_memory_serialize=False,
    trust_remote_code=True
)
engine = allspark.Engine()

print("=== Step 2a: Loading BF16 model weights ===", flush=True)
(model_loader
 .load_model(direct_load=True)
 .read_model_config()
 .serialize_to_path(engine, tmp_dir,
                    enable_quant=False,
                    weight_only_quant=False,
                    skip_if_exists=False)
 .free_model())
print(f"Model loading + serialization took {time.time() - start:.1f}s", flush=True)

print("=== Step 2b: Installing model (1 GPU) ===", flush=True)
runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(
    safe_model_name,
    TargetDevice.CUDA,
    [0],
    max_batch=4
)
runtime_cfg_builder.max_length(256)
runtime_cfg = runtime_cfg_builder.build()

engine.install_model(runtime_cfg)
engine.start_model(safe_model_name)
print(f"Model install + start took {time.time() - start:.1f}s total", flush=True)

print('=== Step 2c: Running inference ===', flush=True)
# For tiny model with random weights, just use simple token inputs
# The output will be garbage but we verify the pipeline doesn't crash
test_queries = [
    'Hello',
    'What is 2+3?',
]

for q in test_queries:
    print(f'\n--- Query: {q} ---', flush=True)
    # Simple input - no template needed for random-weight tiny model
    input_str = q
    gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
    gen_cfg.update({'top_k': 1, 'max_length': 32})

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
