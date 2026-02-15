#!/usr/bin/env python3
"""
Regression checker for DashInfer accuracy benchmarks.

Compares evaluation results against a baseline and reports regressions.
Can be used standalone or imported by run_eval.py.

Usage:
    # Create a baseline from evaluation results
    python check_regression.py create-baseline \
        --results eval_results/eval_standard_20260214.json \
        --output baselines/cpu_qwen2.5_7b.json

    # Check for regressions
    python check_regression.py check \
        --baseline baselines/cpu_qwen2.5_7b.json \
        --results eval_results/eval_standard_20260215.json

 Copyright (c) 2025-2026 DashInfer Team.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


# Default regression thresholds.
# For "higher is better" metrics (accuracy, exact_match), we check for drops.
# For "lower is better" metrics (perplexity, word_perplexity), we check for increases.
DEFAULT_THRESHOLDS = {
    # Metric name pattern → (direction, max_allowed_change)
    # direction: "higher_is_better" or "lower_is_better"
    # max_allowed_change: absolute for accuracy-like, relative for perplexity-like

    # Accuracy-based metrics (higher is better, absolute threshold)
    "acc": ("higher_is_better", 0.03),
    "acc_norm": ("higher_is_better", 0.03),
    "exact_match": ("higher_is_better", 0.02),

    # Perplexity-based metrics (lower is better, relative threshold)
    "word_perplexity": ("lower_is_better", 0.02),
    "byte_perplexity": ("lower_is_better", 0.02),
    "bits_per_byte": ("lower_is_better", 0.02),
}


def get_threshold_for_metric(metric_name: str) -> Optional[Tuple[str, float]]:
    """
    Get the regression threshold for a given metric name.

    Args:
        metric_name: Full metric name (e.g., "acc,none", "word_perplexity,none")

    Returns:
        (direction, threshold) or None if no threshold is defined.
    """
    # Strip lm-eval suffix like ",none"
    clean_name = metric_name.split(",")[0]

    for pattern, threshold_info in DEFAULT_THRESHOLDS.items():
        if pattern in clean_name:
            return threshold_info

    return None


def check_regression(
    baseline_path: str,
    results_path: str,
    custom_thresholds: Optional[Dict] = None,
    verbose: bool = True,
) -> bool:
    """
    Check evaluation results against a baseline for regressions.

    Args:
        baseline_path: Path to the baseline JSON file.
        results_path: Path to the evaluation results JSON file.
        custom_thresholds: Optional dict of custom thresholds to override defaults.
        verbose: Whether to print detailed comparison.

    Returns:
        True if all checks pass, False if any regression is detected.
    """
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    baseline_results = baseline.get("results", {})
    eval_results = results.get("results", {})

    thresholds = DEFAULT_THRESHOLDS.copy()
    if custom_thresholds:
        thresholds.update(custom_thresholds)

    # Also load per-metric thresholds from baseline if present
    baseline_thresholds = baseline.get("thresholds", {})

    all_passed = True
    checks_run = 0
    checks_passed = 0

    if verbose:
        print(f"\n{'='*70}")
        print("Regression Check Report")
        print(f"{'='*70}")
        print(f"Baseline: {baseline_path}")
        print(f"Results:  {results_path}")
        print(f"{'='*70}\n")

    for task_name in baseline_results:
        if task_name not in eval_results:
            if verbose:
                print(f"  WARNING: Task '{task_name}' missing from results")
            continue

        for metric_name, baseline_value in baseline_results[task_name].items():
            if not isinstance(baseline_value, (int, float)):
                continue

            if metric_name not in eval_results[task_name]:
                continue

            current_value = eval_results[task_name][metric_name]
            if not isinstance(current_value, (int, float)):
                continue

            # Determine threshold
            threshold_info = get_threshold_for_metric(metric_name)
            if threshold_info is None:
                continue

            direction, threshold = threshold_info

            # Check for per-metric threshold override in baseline
            override_key = f"{task_name}/{metric_name}"
            if override_key in baseline_thresholds:
                threshold = baseline_thresholds[override_key]

            checks_run += 1

            # Compute difference
            if direction == "higher_is_better":
                # For accuracy: regression if current < baseline - threshold
                diff = current_value - baseline_value
                regression = diff < -threshold
                diff_str = f"{diff:+.4f} (threshold: -{threshold:.4f})"
            else:
                # For perplexity: regression if current > baseline * (1 + threshold)
                if baseline_value == 0:
                    regression = current_value > 0
                    diff_str = f"baseline=0, current={current_value}"
                else:
                    relative_change = (current_value - baseline_value) / abs(baseline_value)
                    regression = relative_change > threshold
                    diff_str = f"{relative_change:+.2%} (threshold: +{threshold:.2%})"

            status = "FAIL" if regression else "PASS"
            if regression:
                all_passed = False
            else:
                checks_passed += 1

            if verbose:
                clean_metric = metric_name.split(",")[0]
                print(f"  [{status}] {task_name}/{clean_metric}")
                print(f"         baseline={baseline_value:.4f}  current={current_value:.4f}  {diff_str}")

    if verbose:
        print(f"\n{'='*70}")
        print(f"Summary: {checks_passed}/{checks_run} checks passed")
        if all_passed:
            print("Status: ALL PASSED")
        else:
            print("Status: REGRESSION DETECTED")
        print(f"{'='*70}\n")

    return all_passed


def create_baseline(results_path: str, output_path: str, description: str = ""):
    """
    Create a baseline file from evaluation results.

    Args:
        results_path: Path to the evaluation results JSON file.
        output_path: Path to write the baseline JSON file.
        description: Optional description for the baseline.
    """
    with open(results_path) as f:
        results = json.load(f)

    baseline = {
        "description": description,
        "source": results_path,
        "config": results.get("config", {}),
        "results": results.get("results", {}),
        "thresholds": {},
    }

    # Pre-populate thresholds from defaults
    for task_name, metrics in baseline["results"].items():
        for metric_name in metrics:
            threshold_info = get_threshold_for_metric(metric_name)
            if threshold_info is not None:
                direction, threshold = threshold_info
                baseline["thresholds"][f"{task_name}/{metric_name}"] = threshold

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline created: {output_path}")
    print(f"Tasks: {list(baseline['results'].keys())}")
    print(f"Thresholds defined: {len(baseline['thresholds'])}")
    print(f"\nYou can edit the thresholds in the baseline file to tune sensitivity.")


def main():
    parser = argparse.ArgumentParser(
        description="DashInfer accuracy regression checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    check_parser = subparsers.add_parser(
        "check", help="Check results against a baseline"
    )
    check_parser.add_argument("--baseline", type=str, required=True,
                              help="Path to baseline JSON file")
    check_parser.add_argument("--results", type=str, required=True,
                              help="Path to evaluation results JSON file")

    # Create-baseline command
    create_parser = subparsers.add_parser(
        "create-baseline", help="Create a baseline from evaluation results"
    )
    create_parser.add_argument("--results", type=str, required=True,
                               help="Path to evaluation results JSON file")
    create_parser.add_argument("--output", type=str, required=True,
                               help="Output path for the baseline file")
    create_parser.add_argument("--description", type=str, default="",
                               help="Description for the baseline")

    args = parser.parse_args()

    if args.command == "check":
        passed = check_regression(args.baseline, args.results, verbose=True)
        sys.exit(0 if passed else 1)
    elif args.command == "create-baseline":
        create_baseline(args.results, args.output, args.description)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
