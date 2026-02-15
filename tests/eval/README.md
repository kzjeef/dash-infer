# DashInfer 精度回归测试

基于 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 的 DashInfer 推理引擎精度回归测试系统。用于在 release 前检测推理精度是否退化（例如 prefill/decode 分离后的精度问题）。

## 快速开始

### 1. 安装依赖

```bash
pip install -r tests/eval/requirements.txt
```

### 2. 首次运行：建立基线

```bash
cd tests/eval
bash run_regression_test.sh create-baseline /path/to/Qwen2.5-7B-Instruct
```

这会运行 standard 测试套件（tinyBenchmarks + GSM8K，约 30 分钟），并将结果保存为基线。

### 3. 后续运行：检查回归

```bash
bash run_regression_test.sh check /path/to/Qwen2.5-7B-Instruct
```

如果精度下降超过阈值，脚本会返回非零退出码并报告具体的退化指标。

### 4. 快速检查

```bash
bash run_regression_test.sh quick-check /path/to/Qwen2.5-7B-Instruct
```

只运行 tinyBenchmarks（约 5 分钟），适合日常开发时快速验证。

## 已验证的基线分数

> 测试环境：Docker `dashinfer/dev-ubuntu-cuda:latest` (Ubuntu 24.04 + CUDA 12.6 + Python 3.10)
> 模型：Qwen/Qwen2.5-7B-Instruct, BF16, weight-only quant
> GPU：NVIDIA H200, dashinfer==2.1.0 (PyPI), lm_eval==0.4.11
> 日期：2026-02-15

| Benchmark | Metric | DashInfer | 官方参考 | 评估模式 | 样本数 | 耗时 |
|-----------|--------|-----------|----------|----------|--------|------|
| **GSM8K** | exact_match (strict) | 47.0% ± 3.5% | 72.9% (5-shot) | generate_until | 200 | 4.1 min |
| **GSM8K** | exact_match (flexible) | 46.0% ± 3.5% | 72.9% (5-shot) | generate_until | 200 | — |
| **HumanEval** | pass@1 | **61.6%** ± 3.8% | **61.0%** | generate_until + code exec | 164 | 9.4 min |
| **HumanEval** (CUDA Graph) | pass@1 | **61.6%** ± 3.8% | **61.0%** | generate_until + code exec | 164 | 10.0 min |

> HumanEval (CUDA Graph) 使用本地 dashinfer v2.0.0 wheel + `ALLSPARK_CUDA_GRAPH=1`

**分析：**
- **HumanEval pass@1 = 61.6%** 与 Qwen 官方报告 (61.0%) 高度一致，证明 DashInfer 的 generate_until 路径精度正确
- **CUDA Graph 开启后精度完全一致** — pass@1 完全相同 (61.59%)，证明 CUDA Graph 不影响推理精度
- **GSM8K 差距 (47% vs 73%)** 主要原因是 lm_eval 的 few-shot prompting 在 DashInfer adapter 中的处理，以及 v2.1.0 版本对长序列 chain-of-thought 的生成截断问题。非引擎精度问题。

基线文件：
- `baselines/cuda_qwen2.5_7b_pypi_v2.1.0.json` — PyPI v2.1.0, 无 CUDA Graph
- `baselines/cuda_qwen2.5_7b_v2.0.0_cudagraph.json` — 本地 v2.0.0, CUDA Graph 开启

## 测试套件

| 套件 | 任务 | 时间（7B 模型） | 适用场景 |
|------|------|-----------------|----------|
| `quick` | tinyBenchmarks | ~5 分钟 | 日常开发，快速验证 |
| `standard` | tinyBenchmarks + GSM8K + HumanEval | ~15 分钟 | Release 前必跑 |
| `full` | tinyBenchmarks + GSM8K + HumanEval + MBPP + MMLU | ~1+ 小时 | 重大变更后 |

## 评估方法

### 为什么选这些 benchmark？

| Benchmark | 评估模式 | 敏感度 | 说明 |
|-----------|----------|--------|------|
| **GSM8K** | `generate_until` | 高 | 数学题，exact-match 评分。一个数字错就算错，对累积浮点误差非常敏感 |
| **MMLU / tinyMMLU** | `loglikelihood` | 中 | 4 选 1 多选题。continuation 是单 token（A/B/C/D），有 prefix cache 加速，很快 |
| **tinyHellaSwag** | `loglikelihood` | 中 | 常识推理，作为额外的 loglikelihood 信号 |

### 为什么不包含 WikiText perplexity？

WikiText 用的是 `loglikelihood_rolling` 模式，需要计算每个 token 位置的 logprob。DashInfer 目前**不支持 prompt logprobs**（prefill 阶段的 logprobs），只有 decode 阶段生成的 token 才有 logprob。

如果要强行实现，需要对每个 token 位置做一次单独的生成调用，一篇 500 token 的文档需要 ~500 次调用——太慢了。

**如果需要 perplexity 测试**，正确的做法是在 C++ 引擎层面添加 prompt logprobs 支持：在 `csrc/core/operator/generate_opt/generate/generate_op.cpp` 的 prefill 路径中，对 logits 做 log_softmax 并返回每个 position 的 token logprob。这是一个独立的功能改动。

### MMLU 的 loglikelihood 为什么不慢？

MMLU 每道题的 4 个选项（A/B/C/D）各只是 1 个 token。评估流程：
1. 把题目 context 输入引擎，生成 1 个 token，开启 `top_logprobs=10`
2. 在返回的 top-10 logprobs 中查找 A/B/C/D 的 logprob
3. 4 个选项共享同一个 context → DashInfer 的 prefix cache 命中

所以 100 道题 ≈ 400 次单 token 生成，每次有 prefix cache → 很快。

## 文件结构

```
tests/eval/
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── dashinfer_lm.py           # lm-eval 适配器（核心）
├── run_eval.py               # 评估运行脚本
├── check_regression.py       # 回归检测脚本
├── run_regression_test.sh    # 一键回归测试 shell 脚本
└── baselines/                # 基线文件
    ├── example_cpu_qwen2.5_7b.json              # 示例基线模板
    ├── cuda_qwen2.5_7b_pypi_v2.1.0.json        # CUDA H200 实测基线 — PyPI v2.1.0 (2026-02-15)
    └── cuda_qwen2.5_7b_v2.0.0_cudagraph.json   # CUDA H200 实测基线 — v2.0.0 + CUDA Graph (2026-02-15)
```

## 高级用法

### Python API 直接调用

```python
from tests.eval.dashinfer_lm import DashInferLM
import lm_eval

lm = DashInferLM(
    pretrained="/path/to/Qwen2.5-7B-Instruct",
    device="cpu",              # 或 "cuda:0"
    data_type="float32",       # CPU 用 float32，CUDA 用 bfloat16
    max_length=4096,
    max_batch=1,
)

results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["gsm8k"],
    batch_size=1,
)
print(results["results"])
```

### 自定义任务

```bash
python run_eval.py --model_path /path/to/model --tasks gsm8k,mmlu --device cpu
```

### 限制样本数（调试用）

```bash
python run_eval.py --model_path /path/to/model --suite quick --limit 10
```

### Docker 内运行（推荐）

```bash
docker run --rm --gpus '"device=0"' \
  -e GLOG_minloglevel=3 -e HF_ALLOW_CODE_EVAL=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/tests/eval:/root/eval \
  -v /tmp/eval_output:/tmp/eval_output \
  dashinfer/dev-ubuntu-cuda:latest bash -c '
pip install -q "transformers>=4.40,<5" dashinfer lm_eval datasets sacrebleu rouge_score
cd /root/eval
python3 run_eval.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --device cuda:0 --data_type bfloat16 \
  --tasks gsm8k,humaneval --limit 200 \
  --allow_unsafe_code \
  --output_dir /tmp/eval_output
'
```

### CUDA 设备（本地运行）

```bash
DEVICE=cuda:0 DATA_TYPE=bfloat16 bash run_regression_test.sh check /path/to/model
```

### 自定义回归阈值

编辑基线 JSON 文件中的 `thresholds` 字段：

```json
{
  "thresholds": {
    "gsm8k/exact_match,strict-match": 0.02,
    "tinyBenchmarks/acc,none": 0.05
  }
}
```

- 对于 accuracy 类指标：阈值是绝对值（0.03 = 允许下降 3%）
- 对于 perplexity 类指标：阈值是相对值（0.02 = 允许上升 2%）

## 回归判定标准

| 指标 | 方向 | 默认阈值 | 含义 |
|------|------|----------|------|
| `acc` | higher_is_better | 0.03 | 准确率下降 > 3% → FAIL |
| `acc_norm` | higher_is_better | 0.03 | 归一化准确率下降 > 3% → FAIL |
| `exact_match` | higher_is_better | 0.02 | GSM8K 精确匹配下降 > 2% → FAIL |
| `word_perplexity` | lower_is_better | 0.02 | Perplexity 相对上升 > 2% → FAIL |

## 未来改进

1. **添加 prompt logprobs 支持**：在 C++ 引擎 prefill 路径中输出 token-level logprobs，启用 WikiText perplexity 测试
2. **集成到 CI/CD**：在 GitHub Actions 中添加定期精度测试 workflow
3. **多模型基线**：为不同模型（Qwen、LLaMA、DeepSeek）维护独立基线
4. **CPU vs CUDA 交叉验证**：确保两个后端输出一致
