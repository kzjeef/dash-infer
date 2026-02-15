<div align="center">

[![PyPI](https://img.shields.io/pypi/v/dashinfer)](https://pypi.org/project/dashinfer/)
[![Documentation Status](https://readthedocs.org/projects/dashinfer/badge/?version=latest)](https://dashinfer.readthedocs.io/en/latest/) 

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/modelscope/dash-infer/blob/main/README_CN.md">中文</a>
    </p>
</h4>


</div>


## News
- [2026/02] 🔥 DashInfer v3.0 released! Major new features include CUDA Graph acceleration for decode phase, DeepSeek V3 (671B) support with Multi-Latent Attention (MLA), FP8 (A8W8) quantization on Hopper GPUs, continuous-batch LoRA optimization, and Expert Parallelism (EP) for large MoE models. For more details, please refer to the [release notes](https://dashinfer.readthedocs.io/en/latest/index.html#v3-0-0).

- [2024/12] DashInfer v2.0 released with enhanced GPU (CUDA) support, prefix caching (with GPU & CPU swapping), guided decoding, optimized attention for GQA, a lockless reactor engine, and newly added support for the VLM model (Qwen-VL) and MoE Models.

- [2024/06] DashInfer v1.0 release with x86 & ARMv9 CPU and CPU flash attention support.

# Introduction

Written in C++ runtime, DashInfer aims to deliver production-level implementations highly optimized for various hardware architectures, including CUDA, x86 and ARMv9.

## Main Features
DashInfer is a highly optimized LLM inference engine with the following core features:

- **Lightweight Architecture**: DashInfer requires minimal third-party dependencies and uses static linking for almost all dependency libraries. By providing C++ and Python interfaces, DashInfer can be easily integrated into your existing system.

- **High Precision**: DashInfer has been rigorously tested to ensure accuracy, and is able to provide inference whose accuracy is consistent with PyTorch and other GPU engines (e.g., vLLM).

- **High Performance**: DashInfer employs optimized kernels to provide high-performance LLM serving, as well as lots of standard LLM inference techniques, including:

  - **Continuous Batching**: DashInfer allows for the immediate insertion of new requests and supports streaming outputs.

  - **Paged Attention**: Using our self-developed paged attention technique (which we call *SpanAttention*), we can achieve efficient acceleration of attention operator, combined with int8 and uint4 KV cache quantization, based on highly efficient GEMM and GEMV implementations.

  - **CUDA Graph**: DashInfer supports CUDA Graph capture for the decode phase, significantly reducing kernel launch overhead and improving throughput for small-batch/latency-sensitive scenarios.

  - **Multi-Latent Attention (MLA)**: DashInfer supports the MLA architecture (used in DeepSeek V3) for compressed KV cache, reducing per-token KV cache by ~28x compared to standard multi-head attention.

  - **Prefix Cache**: DashInfer supports highly efficient Prefix Cache for prompts, which accelerates standard LLMs and MultiModal LMs (MMLMs) like Qwen-VL, using both GPU and CPU.

  - **Quantization Support**: Using DashInfer's *InstantQuant* (IQ), weight-only quantization acceleration can be achieved without fine-tuning, improving deployment efficiency. DashInfer also supports FP8 (A8W8) quantization on Hopper GPUs (SM90+) for further performance gains.

  - **LoRA**: DashInfer supports continuous-batch LoRA optimization with dynamic loading/unloading of LoRA adapters at runtime, enabling efficient multi-tenant serving.

  - **Asynchronous Interface**: Request-based asynchronous interfaces offer individual control over generation parameters and request status of each request.

- Supported Models:

  - **Mainstream Open-Source LLMs**: DashInfer supports mainstream open-source LLMs including Qwen (1/1.5/2/2.5/3), LLaMA (2/3), ChatGLM, DeepSeek V3, and more, supporting loading models in the Huggingface format.

  - **MoE Models**: DashInfer supports Mixture-of-Experts models including Qwen2-MoE and DeepSeek V3 (671B, 256 experts), with Expert Parallelism (EP) support for multi-GPU distribution.

  - **MultiModal LMs**: DashInfer supports MultiModal Language Models (MMLMs) including Qwen-VL, Qwen-AL, and Qwen2-VL.

- **OpenAI API Server**: DashInfer provides OpenAI-compatible API server capabilities for both LLM and VLM serving.

- **Multi-Programming-Language API**: Both C++ and Python interfaces are provided. It is possible to extend C++ interface to Java, Rust and other programming languages, via standard cross-language interfaces.


# Supported Hardware and Data Types

## Hardware
- **CUDA GPUs**: Support CUDA Version from 11.4 - 12.9, and supports various CUDA compute architectures like SM70 - SM100 (T4, 3090, 4090, V100, A100, A10, L20, H20, H100, B200). SM100 (B200) is experimental.
- **x86 CPUs**: Hardware support for AVX2 instruction set is required. For Intel's 5th generation Xeon processors (Emerald Rapids), 4th generation Xeon processors (Sapphire Rapids), corresponding to Aliyun's 8th generation ECS instances (e.g., g8i), AMX instructions are used to accelerate calculation.
- **ARMv9 CPU**: Hardware support for SVE instruction set is required. DashInfer supports ARMv9 architecture processors such as Yitian710, corresponding to Aliyun's 8th generation ECS instances (e.g. g8y), and adopts SVE instruction to accelerate calculation.

## Data Types
- **CUDA GPUs**: FP16, BF16, FP8, FP32, Int8(InstantQuant), Int4(InstantQuant)
- **x86 CPU**: FP32, BF16
- **ARM Yitian710 CPU**: FP32, BF16, Int8(InstantQuant)

### Quantization
DashInfer provides various many quantization technology for LLM weight, such as, int{8,4} weight only quantization, int8 activate quantization, and many customized fused kernel to provide best performance on specified device.

To put it simply, models fine-tuned with GPTQ will provide better accuracy, but our InstantQuant (IQ) technique,
which does not require fine-tuning, can offer a faster deployment experience.
Detailed explanations of IQ quantization can be found at the end of this article.

In terms of supported quantization algorithms, DashInfer supports models fine-tuned with GPTQ and dynamic quantization
using the IQ quantization technique in two ways:

- **IntantQuant(IQ)**: DashInfer provides the InstantQuant (IQ) dynamic quantization technique, which does not require fine-tuning and can offer a faster deployment experience. Detailed explanations of IQ quantization can be found at the end of this article.
- **GPTQ**: Models fine-tuned with GPTQ will provide better accuracy, but it requires a fine-tuning step.

The quantization strategies introduced here can be broadly divided into two categories:

- **Weight Only Quantization**: This quantization technique only quantizes and compresses the weights,
  such as storing weights in int8 format, but uses bf16/fp16 for computations. It only reduces memory access requirements, 
  without improving computational performance compared to BF16.
- **Activation Quantization**: This quantization technique not only stores weights in int8 format but also performs low-precision quantized computations (such as int8) during the calculation phase. (Since Nvidia GPUs only have int8 Tensor Cores that can easily maintain precision, this quantization technique can reduce memory access requirements and improve computational performance, making it a more ideal quantization approach. In terms of accuracy, it may have a slight decrease compared to Weight Only quantization, so business data accuracy testing is required.


In terms of quantization granularity, there are two types:

- **Per-Channel**: DashInfer's quantization techniques at least adopt the Per-Channel (also known as Per-Token) quantization granularity, and some also provide Sub-Channel quantization granularity. Generally speaking, Per-Channel quantization can meet most accuracy requirements due to its simple implementation and optimal performance. Only when the accuracy of Per-Channel quantization is insufficient should the Sub-Channel quantization strategy be considered.
- **Sub-Channel**: Compared to Per-Channel quantization, Sub-Channel refers to dividing a channel into N groups, and calculating quantization parameters within each group. This quantization granularity typically provides better accuracy, but due to increased implementation complexity, it comes with many limitations. For example, performance may be slightly slower than Per-Channel quantization, and Activation quantization is difficult to implement Sub-Channel quantization due to computational formula constraints (DashInfer's Activation quantization is all Per-Channel).

# Software Dependencies

## Build Dependencies

DashInfer uses [Conan 2.x](https://conan.io/) to manage C++ third-party dependencies. The main dependencies include:

| Dependency | Version |
|---|---|
| Conan | >= 2.0 |
| protobuf | 3.18.3 |
| gtest | 1.11.0 |
| glog | 0.5.0 |
| pybind11 | 2.13.6 |
| zlib | 1.2.13 |

> Note: Conan 1.x is no longer supported. Please upgrade to Conan 2.x: `pip install "conan>=2.0"`

## Runtime Dependencies

1. **Python**: DashInfer Python package depends on PyTorch and Huggingface Transformers (for safetensors model weight loading). Individual models may have their own dependencies due to HuggingFace interfaces.
2. **C++**: The C++ package statically links all third-party dependencies with hidden symbols, so there are no runtime third-party library dependencies. Official C++ package distribution is provided through Conan 2.x.

# Documentation and Example Code

## Documentation

For the detailed user manual, please refer to the documentation: [Documentation Link](https://dashinfer.readthedocs.io/en/latest/).

### Quick Start:

1. Using API [Python Quick Start](https://dashinfer.readthedocs.io/en/latest/get_started/quick_start_api_py_en.html)
2. LLM OpenAI Server [Quick Start Guide for OpenAI API Server](https://dashinfer.readthedocs.io/en/latest/get_started/quick_start_api_server_en.html)
3. VLM OpenAI Server [VLM Support](https://dashinfer.readthedocs.io/en/latest/vlm/vlm_offline_inference_en.html)

### Feature Introduction:

1. [Prefix Cache](https://dashinfer.readthedocs.io/en/latest/llm/prefix_caching.html)
2. [Guided Decoding](https://dashinfer.readthedocs.io/en/latest/llm/guided_decoding.html)
3. [Engine Config](https://dashinfer.readthedocs.io/en/latest/llm/runtime_config.html)

### Development:

1. [Development Guide](https://dashinfer.readthedocs.io/en/latest/devel/source_code_build_en.html#)
2. [Build From Source](https://dashinfer.readthedocs.io/en/latest/devel/source_code_build_en.html#build-from-source-code)
3. [OP Profiling](https://dashinfer.readthedocs.io/en/latest/devel/source_code_build_en.html#profiling)
4. [Environment Variable](https://dashinfer.readthedocs.io/en/latest/get_started/env_var_options_en.html)

## Code Examples

In `<path_to_dashinfer>/examples` there are examples for C++ and Python interfaces, and please refer to the documentation in `<path_to_dashinfer>/docs/EN` to run the examples.



- [Base GPU Python Example](examples/python/0_basic/cuda/demo_dashinfer_2_0_gpu_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/dash-infer/blob/main/examples/python/0_basic/cuda/demo_dashinfer_2_0_gpu_example.ipynb)
- [Documentation for All Python Examples](docs/EN/examples_python.md)
- [Documentation for C++ Examples](docs/EN/examples_cpp.md)

## Multi-Modal Model(VLMs) Support

The VLM Support in [multimodal](multimodal/) folder, it's a toolkit to support Vision Language Models (VLMs) inference based on the DashInfer engine. It's compatible with the OpenAI Chat Completion API, supporting text and image/video inputs.

## Performance

We have conducted several benchmarks to compare the performance of mainstream LLM inference engines.

### Multi-Modal Model (VLMs)

We compared the performance of Qwen-VL with vllm across various model sizes:

![img_1.png](docs/resources/image/dashinfer-benchmark-vl.png)

Benchmarks were conducted using an A100-80Gx1 for 2B and 7B sizes, and an A100-80Gx4 for the 72B model. For more details, please refer to the [benchmark documentation](https://github.com/modelscope/dash-infer/blob/main/multimodal/tests/README.md).

### Prefix Cache

We evaluated the performance of the prefix cache at different cache hit rates:

![dahsinfer-benchmark-prefix-cache.png](docs/resources/image/dahsinfer-benchmark-prefix-cache.png)

The chart above shows the reduction in TTFT (Time to First Token) with varying PrefixCache hit rates in DashInfer.

![dashinfer-prefix-effect.png](docs/resources/image/dashinfer-prefix-effect.png)

**Test Setup:**  
- **Model:** Qwen2-72B-Instruct  
- **GPU:** 4x A100  
- **Runs:** 20  
- **Batch Size:** 1  
- **Input Tokens:** 4000  
- **Output Tokens:** 1  

### Guided Decoding (JSON Mode)

We compared the guided output (in JSON format) between different engines using the same request with a customized JSON schema (Context Length: 45, Generated Length: 63):

![dashinfer-benchmark-json-mode.png](docs/resources/image/dashinfer-benchmark-json-mode.png)

# Subprojects

1. [HIE-DNN](https://github.com/modelscope/dash-infer/tree/main/HIE-DNN): an operator library for high-performance inference of deep neural network (DNN).
2. [SpanAttention](https://github.com/modelscope/dash-infer/tree/main/span-attention): a high-performance decode-phase attention with paged KV cache for LLM inference on CUDA-enabled devices.

# Citation

The high-performance implementation of DashInfer MoE operator is introduced in [this paper](https://arxiv.org/abs/2501.16103), and DashInfer employs the efficient top-k operator [*RadiK*](https://arxiv.org/abs/2501.14336).
If you find them useful, please feel free to cite these papers:

```bibtex
@misc{dashinfermoe2025,
  title = {Static Batching of Irregular Workloads on GPUs: Framework and Application to Efficient MoE Model Inference}, 
  author = {Yinghan Li and Yifei Li and Jiejing Zhang and Bujiao Chen and Xiaotong Chen and Lian Duan and Yejun Jin and Zheng Li and Xuanyu Liu and Haoyu Wang and Wente Wang and Yajie Wang and Jiacheng Yang and Peiyang Zhang and Laiwen Zheng and Wenyuan Yu},
  year = {2025},
  eprint = {2501.16103},
  archivePrefix = {arXiv},
  primaryClass = {cs.DC},
  url = {https://arxiv.org/abs/2501.16103}
}

@inproceedings{radik2024,
  title = {RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection},
  author = {Li, Yifei and Zhou, Bole and Zhang, Jiejing and Wei, Xuechao and Li, Yinghan and Chen, Yingda},
  booktitle = {Proceedings of the 38th ACM International Conference on Supercomputing},
  year = {2024}
}
```

# Roadmap

## Completed
- [x] GPU Support
- [x] Multi Modal Model support
- [x] Accelerate attention with Flash-Attention
- [x] Expand context length to over 32k
- [x] Support 4-bit quantization
- [x] Support quantized models fine-tuned with GPTQ
- [x] Support MoE architecture
- [x] Guided output: Json Mode
- [x] Prefix Cache: Support GPU Prefix Cache and CPU Swap 
- [x] Quantization: FP8 A8W8 Activation quantization support on CUDA
- [x] LoRA: Continuous Batch LoRA Optimization
- [x] Parallel Context phase and Generation phase within engine
- [x] More effective MoE Operator on GPU
- [x] CUDA Graph: Piecewise CUDA Graph capture for decode phase acceleration
- [x] MLA: Multi-Latent Attention support (DeepSeek V3)
- [x] Expert Parallelism (EP) for large MoE models

## In Progress & Planned

### [Performance Optimization](docs/EN/roadmap_performance.md)
Goal: match vLLM/SGLang throughput on dense 72B (H100) and DeepSeek V3.2 (B200).
Focus: strengthen DeepSeek serving path with MTP/EAGLE and architecture-specific hardening.
- [ ] Chunked Prefill + Unified Scheduler
- [ ] CUDA Graph Full capture for decode phase
- [ ] Multi-Token Prediction (MTP, DeepSeek-first)
- [ ] Speculative Decoding (EAGLE)
- [ ] DP Attention (Data-Parallel Attention for MoE + MLA)
- [ ] FP4 MoE Fused Kernel (Blackwell B200)
- [ ] NSA Kernel Fusion (DeepSeek V3.2 Native Sparse Attention)
- [ ] DeepSeek Architecture Hardening (MLA/MoE/NextN long-context stability)

### [RL Training Integration](docs/EN/roadmap_rl_integration.md)
Goal: enable DashInfer as inference backend for RLHF/GRPO/DPO training (OpenRLHF, veRL, TRL).
Priority: higher than distributed PD + unified KV-cache work.
- [ ] Prompt Logprobs (prefill-stage log probabilities)
- [ ] In-Place Weight Update (hot reload without restart)
- [ ] Sleep/Wake Mode (GPU memory yield for training)
- [ ] Training-Inference Colocation
- [ ] Ray / Distributed Orchestration Integration

### Other
- [ ] Porting to AMD (ROCm) Platform
- [ ] [Infrastructure Upgrade](docs/EN/roadmap_infra_upgrade.md): Flash Attention 3/4 upgrade, CUTLASS upgrade, Docker image modernization, Conan 2.x, Python 3.10+ default

# License

The DashInfer source code is licensed under the Apache 2.0 license, and you can find the full text of the license in the root of the repository.
