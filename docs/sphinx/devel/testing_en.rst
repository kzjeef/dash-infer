Testing and Evaluation
----------------------

This page centralizes all test and evaluation workflows:

- Regression testing (functional correctness gate)
- Accuracy evaluation
- Performance benchmarking/profiling

Regression Testing (Correctness)
================================

CPU backend is maintained as a backup path. Use the regression script for correctness checks (not performance benchmarking):

.. code-block:: bash

  # quick: C++ interface/operator + Python model-level inference
  CPU_REGRESSION_MODEL_PATH=/path/to/hf_model scripts/run_cpu_regression.sh quick

  # full: quick + cpp_model_test
  CPU_REGRESSION_MODEL_PATH=/path/to/hf_model scripts/run_cpu_regression.sh full

Required environment variable:

- ``CPU_REGRESSION_MODEL_PATH``: local HuggingFace model path used by Python model-level CPU regression.

You can also run C++ unit tests directly after build:

.. code-block:: bash

  AS_PLATFORM="x86" AS_BUILD_PACKAGE=OFF ./build.sh
  ./build/bin/cpp_interface_test
  ./build/bin/cpp_operator_test
  ./build/bin/cpp_model_test


Accuracy Evaluation
===================

Recommended path: use the ``lm-evaluation-harness`` adapter under ``tests/eval``.

- Entry: ``tests/eval/README.md``
- Typical usage includes running benchmark suites such as MMLU, GSM8K, and HellaSwag.
- Use this path for model quality regression and version-to-version comparison.

Legacy Qwen-based evaluation scripts are kept under ``tests/eval/legacy`` for backward compatibility only.


Performance Benchmarking and Profiling
======================================

For synthetic throughput/latency benchmarking, use examples under ``examples/python/1_performance``:

.. code-block:: bash

  cd examples/python/1_performance
  python performance_test_qwen_v15.py

For operator-level profiling:

.. code-block:: bash

  export AS_PROFILE=ON
  # run your benchmark/inference
  # then print engine.get_op_profiling_info(model_name)

For GPU timeline profiling, use controlled Nsys flow (see Developer Guide build page for detailed switches and macros):

.. code-block:: bash

  nsys profile -c cudaProfilerApi xxx_benchmark.py


OpenAI Server API Testing
=========================

DashInfer provides OpenAI-compatible API server tests under ``tests/openai_server``.

1) Start server (example with 5th GPU card):

.. code-block:: bash

  CUDA_VISIBLE_DEVICES=4 python -m dashinfer.allspark.openai_server \
    --model-path /path/to/hf_model \
    --served-model-name Qwen2.5-7B-Instruct \
    --device cuda \
    --device-list 0 \
    --host 127.0.0.1 \
    --port 8000

2) Run API compatibility smoke tests:

.. code-block:: bash

  python tests/openai_server/test_openai_server.py \
    --host 127.0.0.1 \
    --port 8000 \
    --model Qwen2.5-7B-Instruct

3) Run quick concurrent benchmark (vLLM-style OpenAI endpoint pressure test):

.. code-block:: bash

  python tests/openai_server/benchmark_openai_api.py \
    --host 127.0.0.1 \
    --port 8000 \
    --model Qwen2.5-7B-Instruct \
    --requests 64 \
    --concurrency 8 \
    --max-tokens 128 \
    --stream


CI Integration Requirements
===========================

The repository uses two workflow layers:

- ``CI PR Checks`` (fast checks for normal PR/MR)
- ``CI Release Validation`` (broader checks for release tags/manual release validation)

PR Trigger Policy
-----------------

``CI PR Checks`` is configured to skip docs-only changes.

Ignored changes for PR-triggered checks include:

- ``docs/**``
- ``README.md``
- ``README_CN.md``
- ``**/*.md``

If needed, maintainers can still run checks manually with ``workflow_dispatch``.

Runner Requirements
-------------------

Some jobs require a self-hosted CUDA runner. Expected runner labels:

- ``self-hosted``
- ``linux``
- ``gpu``
- ``cuda``

If your infrastructure uses different labels, update ``runs-on`` in workflow files accordingly.

Required GitHub Variables
-------------------------

For ``CI PR Checks``:

- ``DASHINFER_QUICK_PERF_CMD``: command for quick performance regression.
- ``DASHINFER_QUICK_MODEL_PATH``: local model path for quick accuracy check.

For ``CI Release Validation``:

- ``DASHINFER_RELEASE_PERF_CMD``: command for release performance validation.
- ``DASHINFER_RELEASE_MODEL_PATH``: local model path for release accuracy regression.
- ``DASHINFER_RELEASE_MODEL_LIST``: comma-separated local model paths for quick multi-model coverage.

Notes
-----

- Accuracy jobs use ``tests/eval/run_regression_test.sh``.
- PR path runs ``quick-check``; release path runs ``standard`` and optionally ``full``.
- Release workflow triggers on ``v*`` tags and manual dispatch.
