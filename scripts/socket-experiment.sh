#!/usr/bin/env bash
# socket-experiment.sh — single- vs dual-socket CPU scaling for the OpenVINO backend
#
# Container CPU affinity must be set via the Python entrypoint below:
# --cpuset-cpus is rejected by crun because the cpuset controller is not
# delegated to the rootless user session. No memory-node binding is applied
# (numactl and --cpuset-mems unavailable), so NUMA placement is first-touch.
#
# Usage:  ./scripts/socket-experiment.sh [model_name] [duration_seconds]
# Env:    GUIDELLM_STREAMS (default 2)  concurrent streams for the client load
#         CONFIG_FILTER (default all)   comma-separated labels to run
#         INTERLEAVE=1                  set MPOL_INTERLEAVE across nodes 0,1
#                                       via ctypes (numactl unavailable);
#                                       result files get an _interleave suffix
# Example: ./scripts/socket-experiment.sh Qwen3.5-0.8B-int4-ov 90
#          INTERLEAVE=1 CONFIG_FILTER=E_dual_24p GUIDELLM_STREAMS=8 ./scripts/socket-experiment.sh
set -uo pipefail

MODEL="${1:-Qwen3.5-0.8B-int4-ov}"
DURATION="${2:-90}"
GUIDELLM_STREAMS="${GUIDELLM_STREAMS:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/socket-experiment"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$RESULTS_DIR"

HOST_PORT="${HOST_PORT:-8085}"
API_URL="http://localhost:${HOST_PORT}"
CONTAINER_NAME="vllm-socket-exp"
IMAGE="quay.io/joopark/vllm-openvino"
MODEL_DIR="$HOME/hf/OpenVINO/$MODEL"
HF_MODEL_ID="OpenVINO/$MODEL"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model not found: $MODEL_DIR" >&2
    exit 1
fi

cleanup() { podman rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

run_config() {
    local label="$1" cpuset="$2" threads="$3"

    if [[ -n "${CONFIG_FILTER:-}" && ",${CONFIG_FILTER}," != *",${label},"* ]]; then
        return 0
    fi

    local suffix=""
    local py="import os, sys; os.sched_setaffinity(0, {${cpuset}})"
    if [[ "${INTERLEAVE:-0}" == "1" ]]; then
        py+="; import ctypes as _c; _l=_c.CDLL('libc.so.6', use_errno=True); assert _l.syscall(238, 3, _c.byref(_c.c_ulong(3)), 3) == 0, 'set_mempolicy failed errno=' + str(_c.get_errno())"
        suffix="_interleave"
    fi
    py+="; os.execvp('vllm', ['vllm', 'serve'] + sys.argv[1:])"

    local out="$RESULTS_DIR/${TIMESTAMP}_${label}${suffix}.txt"

    echo ""
    echo "=== [${label}${suffix}] cpus=$cpuset threads=$threads ==="
    cleanup

    podman run --replace -d --name "$CONTAINER_NAME" \
        -p "${HOST_PORT}:8080" \
        --memory=16g \
        -v "$PROJECT_ROOT/vllm_openvino:/opt/app-root/vllm_openvino:Z" \
        -v "$MODEL_DIR:/models:Z" \
        -e VLLM_OPENVINO_DEVICE=CPU \
        -e VLLM_OPENVINO_CPU_THREADS_NUM="$threads" \
        -e TORCH_COMPILE_DISABLE=1 \
        --entrypoint python3 \
        "$IMAGE" \
        -c "$py" \
        --port=8080 --model /models --max-model-len 4096 \
        --served-model-name "$HF_MODEL_ID" >/dev/null

    echo "  Waiting for server..."
    local ready=false
    for _ in $(seq 1 60); do
        if curl -sf "$API_URL/v1/models" >/dev/null 2>&1; then ready=true; break; fi
        sleep 5
    done
    if [[ "$ready" != true ]]; then
        echo "  FAIL: startup timeout" | tee -a "$out"
        podman logs --tail 40 "$CONTAINER_NAME" 2>&1 | tee -a "$out"
        cleanup
        return 1
    fi

    echo "  Running guidellm for ${DURATION}s..."
    podman run --rm --network host \
        ghcr.io/vllm-project/guidellm:latest run \
        --backend "kind=openai_http,target=${API_URL},model=${HF_MODEL_ID}" \
        --profile "kind=concurrent,streams=${GUIDELLM_STREAMS}" \
        --constraint "kind=max_duration,seconds=${DURATION}" \
        --data "kind=synthetic_text,prompt_tokens=64,output_tokens=64" \
        --disable-console-interactive \
        2>&1 | tee "$out"

    cleanup
}

echo "=== Socket experiment — $(date) ==="
echo "Model     : $HF_MODEL_ID"
echo "Duration  : ${DURATION}s per config"
echo "Results   : $RESULTS_DIR"

# NUMA node0 physical cores (even): 0,2,4,...,22   (12 cores)
# NUMA node1 physical cores (odd):  1,3,5,...,23   (12 cores)
run_config "A_single_8p"  "0,2,4,6,8,10,12,14"                                              "8"
run_config "B_dual_8p"    "0,2,4,6,1,3,5,7"                                                 "8"
run_config "C_single_12p" "0,2,4,6,8,10,12,14,16,18,20,22"                                  "12"
run_config "E_dual_24p"   "0,2,4,6,8,10,12,14,16,18,20,22,1,3,5,7,9,11,13,15,17,19,21,23"   "24"

echo ""
echo "=== Summary ==="
for f in "$RESULTS_DIR"/${TIMESTAMP}_*.txt; do
    [[ -f "$f" ]] || continue
    echo ""
    echo "--- $(basename "$f") ---"
    tail -25 "$f"
done
