#!/bin/bash
set -x
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
LOGFILE=/scratch/workspaces/jiejing/workdir/allspark/dash-infer/model_test_result.log

echo "=== Waiting for wheel build ===" | tee $LOGFILE
# Wait for wheel to appear
while [ ! -f /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/dist/dashinfer-2.0.0-cp310-cp310-linux_x86_64.whl ]; do
    sleep 30
    echo "$(date): waiting for wheel..." >> $LOGFILE
done
echo "$(date): Wheel found!" | tee -a $LOGFILE

# Install
echo "=== Installing wheel ===" | tee -a $LOGFILE
pip install /scratch/workspaces/jiejing/workdir/allspark/dash-infer/python/dist/dashinfer-2.0.0-cp310-cp310-linux_x86_64.whl --force-reinstall --no-deps 2>&1 | tee -a $LOGFILE

# Verify PTX
echo "=== Verifying PTX in installed so ===" | tee -a $LOGFILE
SO_PATH=$(find /home/jzhang/.local/lib/python3.10/site-packages/dashinfer -name "_allspark*.so" | head -1)
cuobjdump $SO_PATH 2>&1 | grep -E "^(Fatbin|arch)" | sort | uniq -c | tee -a $LOGFILE

# Run model test
echo "=== Running model inference test ===" | tee -a $LOGFILE
CUDA_VISIBLE_DEVICES=0 python3 -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.prompt_utils import PromptTemplate
from dashinfer.allspark._allspark import GenerateRequestStatus
import modelscope

modelscope_name = 'qwen/Qwen2.5-7B-Instruct'
model_local_path = modelscope.snapshot_download(modelscope_name)
safe_model_name = str(modelscope_name).replace('/', '_')
tmp_dir = '/tmp/dashinfer_test_output'

model_loader = allspark.HuggingFaceModel(model_local_path, safe_model_name, user_set_data_type='bfloat16', in_memory_serialize=False, trust_remote_code=True)
engine = allspark.Engine()

(model_loader.load_model().read_model_config()
 .serialize_to_path(engine, tmp_dir, enable_quant=False, weight_only_quant=True, skip_if_exists=True)
 .free_model())

runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA, [0], max_batch=4)
runtime_cfg_builder.max_length(128)
runtime_cfg = runtime_cfg_builder.build()
engine.install_model(runtime_cfg)
engine.start_model(safe_model_name)

print('=== Running inference ===', flush=True)
for q in ['Hello, who are you?', 'What is 2+3?']:
    input_str = PromptTemplate.apply_chatml_template(q)
    gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
    gen_cfg.update({'top_k': 1, 'max_length': 64})
    status, handle, queue = engine.start_request_text(safe_model_name, model_loader, input_str, gen_cfg)
    engine.sync_request(safe_model_name, handle)
    generated_elem = queue.Get()
    generated_ids = generated_elem.ids_from_generate
    output_text = model_loader.init_tokenizer().get_tokenizer().decode(generated_ids)
    print(f'Q: {q}')
    print(f'A: {output_text}')
    print(f'Tokens: {len(generated_ids)}')
    print('---')
    engine.release_request(safe_model_name, handle)

engine.stop_model(safe_model_name)
engine.release_model(safe_model_name)
print('=== ALL DONE ===')
" 2>&1 | tee -a $LOGFILE

echo "=== Test finished at $(date) ===" | tee -a $LOGFILE
