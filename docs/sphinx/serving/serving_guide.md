# DashInfer OpenAI API Server

DashInfer provides a built-in OpenAI-compatible API server for both LLM and VLM (Vision Language Model) serving. One command, auto-detects the model type, zero external serving dependencies.

---

## 1. Installation

```bash
# LLM serving only
pip install "dashinfer[serving]"

# LLM + VLM serving
pip install "dashinfer[serving,vlm]"
```

---

## 2. Start Server

### LLM

```bash
# Basic -- model type is auto-detected
dashinfer_serve --model Qwen/Qwen2.5-7B-Instruct

# Multi-GPU (4-way tensor parallelism)
dashinfer_serve --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel 4

# Custom port, max batch, max sequence length
dashinfer_serve --model Qwen/Qwen2.5-7B-Instruct \
    --port 8080 --max-batch 64 --max-length 16384

# With API key authentication
dashinfer_serve --model Qwen/Qwen2.5-7B-Instruct \
    --api-keys sk-key1,sk-key2

# FP16 weights instead of default BF16
dashinfer_serve --model Qwen/Qwen2.5-7B-Instruct \
    --data-type float16
```

### VLM

```bash
# Auto-detected as VLM (requires dashinfer[serving,vlm])
dashinfer_serve --model Qwen/Qwen2-VL-7B-Instruct

# Force VLM mode, use Transformers vision encoder instead of TensorRT
dashinfer_serve --model Qwen/Qwen2-VL-7B-Instruct \
    --mode vlm --vision-engine transformers

# Multi-GPU VLM
dashinfer_serve --model Qwen/Qwen2-VL-72B-Instruct \
    --tensor-parallel 4
```

### Alternative: Python Module

```bash
python -m dashinfer.serving --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

Once the server is running, API documentation is available at `http://localhost:8000/docs`.

---

## 3. LLM Request Examples

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Write a short poem about the sea."}],
    "stream": true
  }'
```

### Using the OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in 3 sentences."},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

### Streaming with Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

stream = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

### With API Key

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-key1" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Health Check and Model List

```bash
# Health check
curl http://localhost:8000/health
# -> {"status":"ok","mode":"llm"}

# List models
curl http://localhost:8000/v1/models
# -> {"object":"list","data":[{"id":"Qwen2.5-7B-Instruct","object":"model",...}]}
```

---

## 4. VLM Request Examples

### Image Understanding

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
          {"type": "text", "text": "What is in this image?"}
        ]
      }
    ],
    "max_tokens": 512
  }'
```

### VLM with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="Qwen2-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

---

## 5. CLI Reference

**Common options:**

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model path or ID |
| `--mode` | `auto` | `auto` / `llm` / `vlm` |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--tensor-parallel` | `1` | Number of GPUs |
| `--max-batch` | `32` | Max concurrent requests |
| `--max-length` | `8192` | Max sequence length |
| `--data-type` | `bfloat16` | `float16` / `bfloat16` / `float32` |
| `--api-keys` | *(none)* | Comma-separated API keys |
| `--log-level` | `info` | `debug` / `info` / `warning` / `error` |

**VLM-specific options** (ignored in LLM mode):

| Argument | Default | Description |
|---|---|---|
| `--vision-engine` | `tensorrt` | `tensorrt` / `transformers` |
| `--min-pixels` | `3136` | Min image resolution (pixels) |
| `--max-pixels` | `12845056` | Max image resolution (pixels) |
| `--quant-type` | *(none)* | `gptq` / `a8w8` / `a16w4` / `a16w8` / `fp8` |

Full help: `dashinfer_serve --help`

---

## 6. API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (returns mode: llm/vlm) |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
| `GET` | `/docs` | Swagger UI interactive documentation |
