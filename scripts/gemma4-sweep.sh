#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/guidellm-results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

API_URL="http://localhost:8082"
CONTAINER_NAME="vllm-gemma-sweep"
IMAGE="quay.io/joopark/vllm-openvino"
MODEL_DIR="$HOME/hf/OpenVINO/gemma-4-E2B-it-int4-ov"
HF_MODEL_ID="OpenVINO/gemma-4-E2B-it-int4-ov"

GUIDELLM_STREAMS="${GUIDELLM_STREAMS:-2}"
GUIDELLM_MAX_SECONDS="${GUIDELLM_MAX_SECONDS:-120}"
GUIDELLM_PROMPT_TOKENS="${GUIDELLM_PROMPT_TOKENS:-64}"
GUIDELLM_OUTPUT_TOKENS="${GUIDELLM_OUTPUT_TOKENS:-64}"

cleanup() {
  podman stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  podman rm   "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

run_config() {
  local label="$1"; shift
  local -a extra_env=("$@")

  echo ""
  echo "=== $label ==="
  cleanup

  local env_args=()
  for e in "${extra_env[@]}"; do env_args+=("-e" "$e"); done

  podman run --replace -d --name "$CONTAINER_NAME" \
    -p 8082:8080 --memory=16g \
    -v "$PROJECT_ROOT/vllm_openvino:/opt/app-root/vllm_openvino:z" \
    -v "$MODEL_DIR:/models:Z" \
    -e VLLM_OPENVINO_DEVICE=CPU \
    -e TORCH_COMPILE_DISABLE=1 \
    -e VLLM_OPENVINO_CPU_THREADS_NUM=8 \
    "${env_args[@]}" \
    "$IMAGE" \
    --port=8080 --model /models --max-model-len 4096 \
    --served-model-name "$HF_MODEL_ID" >/dev/null

  echo "  Waiting for startup..."
  local ready=false
  for _ in $(seq 1 60); do
    if curl -sf "$API_URL/v1/models" >/dev/null 2>&1; then
      ready=true; break
    fi
    if ! podman ps --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
      echo "  FAIL: container exited"; podman logs "$CONTAINER_NAME" 2>&1 | tail -20
      return 1
    fi
    sleep 5
  done
  if [[ "$ready" != true ]]; then
    echo "  FAIL: startup timeout"; podman logs "$CONTAINER_NAME" 2>&1 | tail -20
    return 1
  fi
  echo "  Server ready"

  local result_file="$RESULTS_DIR/${TIMESTAMP}_gemma4_${label}.txt"
  podman run --rm --network host \
    ghcr.io/vllm-project/guidellm:latest run \
    --backend "kind=openai_http,target=${API_URL},model=${HF_MODEL_ID}" \
    --profile "kind=concurrent,streams=${GUIDELLM_STREAMS}" \
    --constraint "kind=max_duration,seconds=${GUIDELLM_MAX_SECONDS}" \
    --data "kind=synthetic_text,prompt_tokens=${GUIDELLM_PROMPT_TOKENS},output_tokens=${GUIDELLM_OUTPUT_TOKENS}" \
    --disable-console-interactive \
    2>&1 | tee "$result_file" || true
  echo "  Saved: $result_file"
  cleanup
}

main() {
  mkdir -p "$RESULTS_DIR"
  echo "=== Gemma-4 sweep — $(date) ==="

  run_config "baseline"           # defaults: FAST_SAMPLER=1
  run_config "nofast" VLLM_OPENVINO_FAST_SAMPLER=0
  run_config "latency" VLLM_OPENVINO_PERFORMANCE_MODE=LATENCY

  echo ""
  echo "=== Done ==="
  ls -lh "$RESULTS_DIR"/*"${TIMESTAMP}"* 2>/dev/null
}

main "$@"
