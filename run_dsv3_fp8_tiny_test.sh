#!/bin/bash
set -x
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
LOG=/scratch/workspaces/jiejing/workdir/allspark/dash-infer/dsv3_fp8_tiny_test.log
echo "=== DeepSeek V3 FP8 A8W8 Tiny Test at $(date) ===" | tee $LOG

# Install wheel + overlay Python files
WHL=/scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/dist/dashinfer-2.0.0-cp310-cp310-linux_x86_64.whl
pip install "$WHL" --force-reinstall --no-deps 2>&1 | tee -a $LOG
SITE_PKG=$(python3 -c "import dashinfer; import os; print(os.path.dirname(dashinfer.__file__))")
for f in model/deepseek_v3.py model/__init__.py model/model_base.py; do
    cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/$f "$SITE_PKG/allspark/$f" 2>&1
done
for f in model_config.py engine_utils.py model_loader.py; do
    cp -v /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/pyhie/allspark/$f "$SITE_PKG/allspark/$f" 2>&1
done

cd /scratch/workspaces/jiejing/workdir/allspark/dash-infer

timeout 300 python3 << 'PYEOF' 2>&1 | tee -a $LOG
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark._allspark import AsStatus

model_local_path = '/scratch/workspaces/jiejing/models/tiny-dsv3'
safe_model_name = 'DeepSeek_v3'
tmp_dir = '/scratch/workspaces/jiejing/tmp/dashinfer_dsv3_tiny_fp8_output'

print("=== Loading tiny model with FP8 A8W8 quantization ===", flush=True)
start = time.time()

model_loader = allspark.HuggingFaceModel(
    model_local_path,
    safe_model_name,
    user_set_data_type='bfloat16',
    in_memory_serialize=False,
    trust_remote_code=True
)
engine = allspark.Engine()

# Use FP8 A8W8 quantization via instant_quant
fp8_quant_config = {
    "quant_method": "instant_quant",
    "weight_format": "fp8_e4m3",
    "compute_method": "activate_quant",
    "activate_format": "fp8_e4m3",
}

print("=== Serializing with FP8 A8W8 ===", flush=True)
(model_loader
 .load_model(direct_load=True)
 .read_model_config()
 .serialize_to_path(engine, tmp_dir,
                    enable_quant=True,
                    weight_only_quant=False,
                    customized_quant_config=fp8_quant_config,
                    skip_if_exists=False)
 .free_model())
print(f"Model loading + serialization took {time.time() - start:.1f}s", flush=True)

print("=== Installing model (1 GPU) ===", flush=True)
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

print('=== Running FP8 inference ===', flush=True)
tokenizer = model_loader.init_tokenizer().get_tokenizer()
for q in ['Hello', 'What is 2+3?']:
    print(f'\n--- Query: {q} ---', flush=True)
    # Apply chat template for correct prompt formatting
    messages = [{"role": "user", "content": q}]
    input_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    print(f'Formatted: {repr(input_str[:100])}', flush=True)

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
print('\n=== FP8 A8W8 TEST PASSED ===')
PYEOF

TEST_EXIT=$?
if [ $TEST_EXIT -ne 0 ]; then
    echo "=== TEST FAILED with exit code $TEST_EXIT ===" | tee -a $LOG
else
    echo "=== Test finished at $(date) ===" | tee -a $LOG
fi
