# OpenAI Server Tests

This directory provides baseline tests and benchmark scripts for
`dashinfer.allspark.openai_server`.

## 1) Start server (example: use 5th GPU card)

```bash
CUDA_VISIBLE_DEVICES=4 python -m dashinfer.allspark.openai_server \
  --model-path /home/jzhang/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --served-model-name Qwen2.5-7B-Instruct \
  --device cuda \
  --device-list 0 \
  --host 127.0.0.1 \
  --port 8000
```

## 2) Run API compatibility smoke tests

```bash
python tests/openai_server/test_openai_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen2.5-7B-Instruct
```

Enable strict tool-call assertion:

```bash
python tests/openai_server/test_openai_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen2.5-7B-Instruct \
  --strict-tool-call
```

## 3) Run concurrent benchmark (vLLM-style)

```bash
python tests/openai_server/benchmark_openai_api.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen2.5-7B-Instruct \
  --requests 64 \
  --concurrency 8 \
  --max-tokens 128 \
  --stream
```
