#!/bin/bash
# ============================================================================
# DashInfer Accuracy Regression Test
#
# Run this before each release to verify model precision has not regressed.
#
# Usage:
#   # First run: create a baseline
#   bash run_regression_test.sh create-baseline /path/to/model
#
#   # Subsequent runs: check for regressions
#   bash run_regression_test.sh check /path/to/model
#
#   # Quick check (faster, less comprehensive)
#   bash run_regression_test.sh quick-check /path/to/model
#
# Environment variables:
#   DEVICE          - "cpu" (default) or "cuda:0"
#   DATA_TYPE       - "float32" (default) or "bfloat16"
#   MAX_LENGTH      - max sequence length (default: 4096)
#   SUITE           - benchmark suite: "quick", "standard" (default), "full"
#   BASELINE_DIR    - directory for baseline files (default: baselines/)
#   EVAL_OUTPUT_DIR - directory for results (default: eval_results/)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
DEVICE="${DEVICE:-cpu}"
DATA_TYPE="${DATA_TYPE:-float32}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
SUITE="${SUITE:-standard}"
BASELINE_DIR="${BASELINE_DIR:-baselines}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-eval_results}"

# Derive baseline filename from device and data type
get_baseline_name() {
    local model_name="$(basename "$1" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
    echo "${BASELINE_DIR}/${DEVICE}_${DATA_TYPE}_${model_name}.json"
}

usage() {
    echo "Usage: $0 <command> <model_path>"
    echo ""
    echo "Commands:"
    echo "  create-baseline  Run evaluation and save as baseline"
    echo "  check            Run evaluation and compare against baseline"
    echo "  quick-check      Quick check (tinyBenchmarks only)"
    echo ""
    echo "Environment variables:"
    echo "  DEVICE=$DEVICE  DATA_TYPE=$DATA_TYPE  SUITE=$SUITE"
    exit 1
}

run_eval() {
    local model_path="$1"
    local suite="$2"
    local extra_args="${3:-}"

    echo "============================================================"
    echo "DashInfer Accuracy Regression Test"
    echo "============================================================"
    echo "Model:     $model_path"
    echo "Device:    $DEVICE"
    echo "Data type: $DATA_TYPE"
    echo "Suite:     $suite"
    echo "============================================================"
    echo ""

    python3 run_eval.py \
        --model_path "$model_path" \
        --device "$DEVICE" \
        --data_type "$DATA_TYPE" \
        --max_length "$MAX_LENGTH" \
        --suite "$suite" \
        --output_dir "$EVAL_OUTPUT_DIR" \
        $extra_args
}

# ---- Main ----

if [ $# -lt 2 ]; then
    usage
fi

COMMAND="$1"
MODEL_PATH="$2"

case "$COMMAND" in
    create-baseline)
        # Run evaluation and create baseline
        run_eval "$MODEL_PATH" "$SUITE"

        # Find the latest result file
        LATEST_RESULT=$(ls -t "$EVAL_OUTPUT_DIR"/eval_*.json 2>/dev/null | head -1)
        if [ -z "$LATEST_RESULT" ]; then
            echo "ERROR: No evaluation results found"
            exit 1
        fi

        BASELINE_FILE=$(get_baseline_name "$MODEL_PATH")
        python3 check_regression.py create-baseline \
            --results "$LATEST_RESULT" \
            --output "$BASELINE_FILE" \
            --description "Baseline for $(basename "$MODEL_PATH") on $DEVICE ($DATA_TYPE)"

        echo ""
        echo "Baseline saved to: $BASELINE_FILE"
        echo "Edit the thresholds in this file to tune regression sensitivity."
        ;;

    check)
        BASELINE_FILE=$(get_baseline_name "$MODEL_PATH")
        if [ ! -f "$BASELINE_FILE" ]; then
            echo "ERROR: Baseline file not found: $BASELINE_FILE"
            echo "Run '$0 create-baseline $MODEL_PATH' first."
            exit 1
        fi

        run_eval "$MODEL_PATH" "$SUITE" "--baseline $BASELINE_FILE"
        ;;

    quick-check)
        BASELINE_FILE=$(get_baseline_name "$MODEL_PATH")
        SUITE="quick"
        if [ ! -f "$BASELINE_FILE" ]; then
            echo "WARNING: No baseline file found at $BASELINE_FILE"
            echo "Running evaluation without baseline comparison."
            run_eval "$MODEL_PATH" "quick"
        else
            run_eval "$MODEL_PATH" "quick" "--baseline $BASELINE_FILE"
        fi
        ;;

    *)
        echo "Unknown command: $COMMAND"
        usage
        ;;
esac
