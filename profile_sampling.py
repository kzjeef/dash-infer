#!/usr/bin/env python3
"""Profile sampling kernels with nsys."""
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch, torch.utils.dlpack as dlpack
from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.runtime_config import AsModelRuntimeConfigBuilder
from transformers import AutoTokenizer

safe = 'qwen_Qwen2.5-7B-Instruct'
tmp_dir = '/tmp/dashinfer_test_output'
tok = AutoTokenizer.from_pretrained(
    os.path.expanduser('~/.cache/modelscope/hub/models/qwen/Qwen2___5-7B-Instruct'),
    trust_remote_code=True)

engine = allspark.Engine()
cfg = (AsModelRuntimeConfigBuilder()
    .model_name(safe).model_dir(tmp_dir, safe)
    .compute_unit(TargetDevice.CUDA, [0], 0)
    .max_length(256).max_batch(1).build())
engine.install_model(cfg)
engine.start_model(safe)

msgs = [{'role': 'user', 'content': 'What is 2+3?'}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok.encode(text)
t = torch.LongTensor([ids]).cpu().contiguous()
dl = dlpack.to_dlpack(t)

# nsys will capture this region
status, handle, queue = engine.start_request(safe, {'input_ids': dl},
    {'top_k': 1, 'max_length': 32})
engine.sync_request(safe, handle)
gen = queue.Get()
print(f"Generated {len(gen.ids_from_generate)} tokens")
engine.release_request(safe, handle)

engine.stop_model(safe)
engine.release_model(safe)
