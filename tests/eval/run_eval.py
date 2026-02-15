#!/usr/bin/env python3
"""
Run lm-evaluation-harness benchmarks against DashInfer.

This script provides a convenient wrapper around lm-evaluation-harness
with DashInfer as the inference backend. It supports predefined benchmark
suites designed for precision regression testing.

Usage:
    # Quick precision regression check (~5 min with 7B model)
    python run_eval.py --model_path /path/to/qwen2.5-7b --suite quick

    # Standard regression suite (~30 min)
    python run_eval.py --model_path /path/to/qwen2.5-7b --suite standard

    # Full evaluation (~1+ hour)
    python run_eval.py --model_path /path/to/qwen2.5-7b --suite full

    # Custom tasks
    python run_eval.py --model_path /path/to/qwen2.5-7b --tasks gsm8k,mmlu

    # With baseline comparison
    python run_eval.py --model_path /path/to/qwen2.5-7b --suite standard \
        --baseline baselines/cpu_qwen2.5_7b.json

 Copyright (c) 2025-2026 DashInfer Team.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Predefined benchmark suites for regression testing
BENCHMARK_SUITES = {
    # Quick sanity check: catches catastrophic failures (~5 min)
    # Uses tinyBenchmarks which includes tinyMMLU etc.
    # loglikelihood tasks in tinyBenchmarks use single-token continuations → fast
    "quick": {
        "tasks": "tinyBenchmarks",
        "description": "Quick sanity check (~5 min): tinyBenchmarks (tinyMMLU, tinyHellaSwag, etc.)",
        "num_fewshot": None,  # use task defaults
        "limit": None,
    },

    # Standard regression suite: ~20 min
    # - tinyBenchmarks (loglikelihood, single-token) - catches weight loading / catastrophic issues
    # - GSM8K 200 examples (generate_until) - sensitive to accumulated FP errors in decode path
    # - HumanEval (generate_until + code exec) - extremely sensitive: one wrong token = test failure
    # Covers 3 dimensions: knowledge (MMLU), math (GSM8K), code (HumanEval)
    "standard": {
        "tasks": "tinyBenchmarks,gsm8k,humaneval",
        "description": "Standard regression suite (~20 min): tinyBenchmarks + GSM8K(200) + HumanEval",
        "num_fewshot": None,
        "limit": {"gsm8k": 200},  # limit GSM8K to 200 examples for speed
    },

    # Full evaluation: comprehensive but slower (~1+ hour)
    # - Full GSM8K (1319 examples)
    # - Full MMLU
    # - HumanEval + MBPP
    # Note: WikiText perplexity (loglikelihood_rolling) is NOT included because
    # DashInfer lacks prompt logprobs, making it extremely slow (O(N) calls per
    # token). To add perplexity testing, first implement prompt logprobs in the
    # C++ engine (prefill path of generate_op).
    "full": {
        "tasks": "tinyBenchmarks,gsm8k,humaneval,mbpp,mmlu",
        "description": "Full evaluation (~1+ hour): tinyBenchmarks + GSM8K + HumanEval + MBPP + MMLU",
        "num_fewshot": None,
        "limit": None,
    },
}


def run_evaluation(args):
    """Run the lm-evaluation-harness with DashInfer backend."""
    import lm_eval

    # Add our module path so lm_eval can find the adapter
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    from dashinfer_lm import DashInferLM

    # Determine tasks
    if args.suite:
        suite = BENCHMARK_SUITES[args.suite]
        tasks_str = suite["tasks"]
        print(f"\n{'='*60}")
        print(f"Running: {suite['description']}")
        print(f"{'='*60}\n")
    elif args.tasks:
        tasks_str = args.tasks
    else:
        print("Error: specify either --suite or --tasks")
        sys.exit(1)

    tasks = tasks_str.split(",")

    # Initialize DashInfer model
    print(f"Initializing DashInfer with model: {args.model_path}")
    print(f"Device: {args.device}, Data type: {args.data_type}")
    print(f"Max length: {args.max_length}, Max batch: {args.max_batch}")

    model_kwargs = {
        "pretrained": args.model_path,
        "device": args.device,
        "data_type": args.data_type,
        "max_length": args.max_length,
        "max_batch": args.max_batch,
        "max_gen_toks": args.max_gen_toks,
        "weight_only_quant": args.weight_only_quant,
    }

    if args.model_output_dir:
        model_kwargs["model_output_dir"] = args.model_output_dir

    lm = DashInferLM(**model_kwargs)

    # Run evaluation
    start_time = time.time()

    eval_kwargs = {
        "model": lm,
        "tasks": tasks,
        "batch_size": args.batch_size,
    }

    if args.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = args.num_fewshot

    # Handle limit: CLI --limit overrides suite-level per-task limits
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit
    elif args.suite and BENCHMARK_SUITES[args.suite].get("limit"):
        # Suite-level per-task limits (e.g., {"gsm8k": 200})
        suite_limit = BENCHMARK_SUITES[args.suite]["limit"]
        if isinstance(suite_limit, dict):
            # lm_eval supports per-task limits via dict
            eval_kwargs["limit"] = suite_limit
        else:
            eval_kwargs["limit"] = suite_limit

    # Allow unsafe code execution for tasks like HumanEval/MBPP
    if args.allow_unsafe_code or os.environ.get("HF_ALLOW_CODE_EVAL") == "1":
        eval_kwargs["confirm_run_unsafe_code"] = True


    results = lm_eval.simple_evaluate(**eval_kwargs)

    elapsed = time.time() - start_time

    # Process results
    print(f"\n{'='*60}")
    print(f"Evaluation completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    # Print summary table
    print_results_table(results)

    # Save results
    output = {
        "config": {
            "model_path": args.model_path,
            "device": args.device,
            "data_type": args.data_type,
            "max_length": args.max_length,
            "suite": args.suite,
            "tasks": tasks_str,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
        },
        "results": {},
    }

    for task_name, task_results in results.get("results", {}).items():
        output["results"][task_name] = {}
        for metric, value in task_results.items():
            # Filter to keep only numeric metrics
            if isinstance(value, (int, float)):
                output["results"][task_name][metric] = value

    # Save to file
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"eval_{args.suite or 'custom'}_{timestamp}.json"
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Compare with baseline if provided
    if args.baseline:
        from check_regression import check_regression
        passed = check_regression(args.baseline, output_file, verbose=True)
        if not passed:
            print("\n*** REGRESSION DETECTED ***")
            sys.exit(1)
        else:
            print("\n*** ALL CHECKS PASSED ***")

    return output


def print_results_table(results):
    """Print a formatted summary table of evaluation results."""
    task_results = results.get("results", {})
    if not task_results:
        print("No results to display.")
        return

    # Collect all metrics
    rows = []
    for task_name, metrics in task_results.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                # Format metric name nicely
                display_name = metric_name.replace(",none", "").replace(",", " ")
                if isinstance(value, float):
                    rows.append((task_name, display_name, f"{value:.4f}"))
                else:
                    rows.append((task_name, display_name, str(value)))

    if not rows:
        print("No numeric results to display.")
        return

    # Print table
    max_task = max(len(r[0]) for r in rows)
    max_metric = max(len(r[1]) for r in rows)
    max_value = max(len(r[2]) for r in rows)

    header = f"{'Task':<{max_task}}  {'Metric':<{max_metric}}  {'Value':>{max_value}}"
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)
    prev_task = None
    for task, metric, value in rows:
        display_task = task if task != prev_task else ""
        print(f"{display_task:<{max_task}}  {metric:<{max_metric}}  {value:>{max_value}}")
        prev_task = task
    print(separator)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM accuracy benchmarks with DashInfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Predefined suites:
  quick      tinyBenchmarks (~5 min) - catches catastrophic failures
  standard   tinyBenchmarks + GSM8K (~30 min) - good regression coverage
  full       tinyBenchmarks + GSM8K + WikiText (~1+ hour) - comprehensive

Examples:
  # Quick check on CPU with Qwen2.5-7B
  python run_eval.py --model_path /path/to/Qwen2.5-7B-Instruct --suite quick --device cpu

  # Standard check on CUDA
  python run_eval.py --model_path /path/to/Qwen2.5-7B-Instruct --suite standard --device cuda:0

  # Compare with baseline
  python run_eval.py --model_path /path/to/model --suite standard \\
      --baseline baselines/cpu_qwen2.5_7b.json
        """,
    )

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model (local or model ID)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu', 'cuda:0', 'cuda:0,1' (default: cpu)")
    parser.add_argument("--data_type", type=str, default="float32",
                        choices=["float32", "bfloat16", "float16"],
                        help="Weight data type (default: float32)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length (default: 4096)")
    parser.add_argument("--max_batch", type=int, default=1,
                        help="Maximum engine batch size (default: 1)")
    parser.add_argument("--max_gen_toks", type=int, default=512,
                        help="Max tokens to generate per request (default: 512)")
    parser.add_argument("--weight_only_quant", action="store_true", default=True,
                        help="Enable weight-only quantization (default: True)")
    parser.add_argument("--no_weight_quant", action="store_true",
                        help="Disable weight-only quantization")
    parser.add_argument("--model_output_dir", type=str, default=None,
                        help="Directory for serialized model files (default: temp dir)")

    # Evaluation arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--suite", type=str, choices=BENCHMARK_SUITES.keys(),
                       help="Predefined benchmark suite")
    group.add_argument("--tasks", type=str,
                       help="Comma-separated list of lm-eval task names")

    parser.add_argument("--num_fewshot", type=int, default=None,
                        help="Number of few-shot examples (default: task-specific)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per task (for testing)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size (default: 1)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory for results (default: eval_results)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline JSON for regression comparison")
    parser.add_argument("--allow_unsafe_code", action="store_true",
                        help="Allow execution of model-generated code (for HumanEval/MBPP)")

    args = parser.parse_args()

    if args.no_weight_quant:
        args.weight_only_quant = False

    if not args.suite and not args.tasks:
        parser.error("Either --suite or --tasks must be specified")

    run_evaluation(args)


if __name__ == "__main__":
    main()
