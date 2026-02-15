## Quick Start Guide for OpenAI API Server

DashInfer provides a built-in OpenAI-compatible API server for both LLM and VLM models. The model type is auto-detected from the HuggingFace config.

### 1. Install

```shell
# LLM serving
pip install "dashinfer[serving]"

# LLM + VLM serving
pip install "dashinfer[serving,vlm]"
```

### 2. Start LLM Server

```shell
dashinfer_serve --model Qwen/Qwen2.5-7B-Instruct

# Multi-GPU
dashinfer_serve --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel 4

# Custom port and settings
dashinfer_serve --model Qwen/Qwen2.5-7B-Instruct \
    --port 8080 --max-batch 64 --max-length 16384
```

Or via Python module:

```shell
python -m dashinfer.serving --model Qwen/Qwen2.5-7B-Instruct
```

### 3. Start VLM Server

```shell
# Auto-detected as VLM (requires dashinfer[serving,vlm])
dashinfer_serve --model Qwen/Qwen2-VL-7B-Instruct

# Force VLM mode with Transformers vision encoder
dashinfer_serve --model Qwen/Qwen2-VL-7B-Instruct \
    --mode vlm --vision-engine transformers
```

### 4. LLM Chat Completion

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello! What is your name?"}
    ]
  }'
```

### 5. Streaming

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Write a short poem."}],
    "stream": true
  }'
```

### 6. VLM Image Understanding

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-VL-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        {"type": "text", "text": "What is in this image?"}
      ]
    }],
    "max_tokens": 512
  }'
```

### 7. Python Client (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# LLM
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing."}],
)
print(response.choices[0].message.content)

# VLM
response = client.chat.completions.create(
    model="Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
            {"type": "text", "text": "Describe this image."},
        ],
    }],
)
print(response.choices[0].message.content)
```

### Available Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check (returns mode: llm/vlm) |
| GET | `/v1/models` | List loaded models |
| POST | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| GET | `/docs` | Swagger UI documentation |

### Server Options

Run `dashinfer_serve --help` for the full list. Key options:

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model path or ID |
| `--mode` | `auto` | `auto` / `llm` / `vlm` |
| `--port` | `8000` | Server port |
| `--tensor-parallel` | `1` | Number of GPUs |
| `--max-batch` | `32` | Max concurrent requests |
| `--max-length` | `8192` | Max sequence length |
| `--api-keys` | *(none)* | Comma-separated API keys |
| `--vision-engine` | `tensorrt` | [VLM] `tensorrt` / `transformers` |

For the full serving guide, see [DashInfer OpenAI API Server](../serving/serving_guide.md).
