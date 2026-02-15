## OpenAI API Chat Server 快速开始

DashInfer 提供项目内原生 OpenAI 兼容 API Server（不依赖 FastChat）。

请先安装可选 server 依赖：

```shell
pip install "dashinfer[server]"
```

### 1. 启动服务

```shell
python -m dashinfer.allspark.openai_server \
  --model-path /path/to/your/model \
  --device cuda \
  --device-list 0 \
  --host 0.0.0.0 \
  --port 8000
```

如果使用 ModelScope 模型名：

```shell
python -m dashinfer.allspark.openai_server \
  --model-path qwen/Qwen2.5-7B-Instruct \
  --use-modelscope \
  --device cuda \
  --device-list 0
```

### 2. 测试普通对话

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "你好，请简单介绍你自己。"}
    ]
  }'
```

### 3. 测试流式输出

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "stream": true,
    "messages": [
      {"role": "user", "content": "写三个简短笑话。"}
    ]
  }'
```

### 4. 测试 Function/Tool Calling

Server 支持 OpenAI 风格 `tools` / `tool_choice`，也兼容旧版 `functions` / `function_call`。
当模型决定调用工具时，返回 `finish_reason=tool_calls`，并携带 `message.tool_calls`。

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "请查询北京天气。"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "根据城市查询天气",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

### 5. 运行自动化接口测试

先启动 server，然后执行：

```shell
python tests/openai_server/test_openai_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen2.5-7B-Instruct
```

如需严格要求返回 `tool_calls`（失败即退出非零）：

```shell
python tests/openai_server/test_openai_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen2.5-7B-Instruct \
  --strict-tool-call
```
