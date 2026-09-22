#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/guidellm-results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

API_URL="http://localhost:8080"
CONTAINER_NAME="vllm-guidellm"
IMAGE="quay.io/joopark/vllm-openvino"
MODEL_BASE="${MODEL_BASE:-$HOME/hf/OpenVINO}"

MODELS=(
  "LFM2.5-8B-A1B-int4-ov"
  "Qwen3.5-2B-int4-ov"
  "gemma-4-E2B-it-int4-ov"
)

GUIDELLM_STREAMS="${GUIDELLM_STREAMS:-2}"
GUIDELLM_MAX_SECONDS="${GUIDELLM_MAX_SECONDS:-180}"
GUIDELLM_PROMPT_TOKENS="${GUIDELLM_PROMPT_TOKENS:-64}"
GUIDELLM_OUTPUT_TOKENS="${GUIDELLM_OUTPUT_TOKENS:-64}"

cleanup_container() {
  podman stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  podman rm   "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

run_model() {
  local model_name="$1"
  local model_dir="$MODEL_BASE/$model_name"
  local hf_model_id="OpenVINO/$model_name"

  if [[ ! -d "$model_dir" ]]; then
    echo "  SKIP: not found: $model_dir"
    return 1
  fi

  echo ""
  echo "=== $model_name ==="
  cleanup_container

  podman run --replace -d --name "$CONTAINER_NAME" \
    -p 8080:8080 --cpus=8 --memory=16g \
    -v "$PROJECT_ROOT/vllm_openvino:/opt/app-root/vllm_openvino" \
    -v "$model_dir:/models:Z" \
    -e VLLM_OPENVINO_DEVICE=CPU \
    -e TORCH_COMPILE_DISABLE=1 \
    "$IMAGE" \
    --port=8080 --model /models --max-model-len 4096 \
    --served-model-name "$hf_model_id" >/dev/null

  echo "  Waiting for server..."
  local ready=false
  for _ in $(seq 1 60); do
    if curl -sf "$API_URL/v1/models" >/dev/null 2>&1; then
      ready=true; break
    fi
    sleep 5
  done

  if [[ "$ready" != true ]]; then
    echo "  FAIL: startup timeout"
    podman logs "$CONTAINER_NAME" 2>&1 | tail -20
    cleanup_container
    return 1
  fi
  echo "  Server ready — model: $hf_model_id"

  local result_file="$RESULTS_DIR/${TIMESTAMP}_${model_name}.txt"

  podman run --rm \
    --network host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    ghcr.io/vllm-project/guidellm:latest run \
    --backend "kind=openai_http,target=${API_URL},model=${hf_model_id}" \
    --profile "kind=concurrent,streams=${GUIDELLM_STREAMS}" \
    --constraint "kind=max_duration,seconds=${GUIDELLM_MAX_SECONDS}" \
    --data "kind=synthetic_text,prompt_tokens=${GUIDELLM_PROMPT_TOKENS},output_tokens=${GUIDELLM_OUTPUT_TOKENS}" \
    --disable-console-interactive \
    2>&1 | tee "$result_file" || true

  echo "  Saved: $result_file"

  cleanup_container
}

main() {
  mkdir -p "$RESULTS_DIR"

  echo "=== guidellm Benchmark — $(date) ==="
  echo "Profile : concurrent @ streams=${GUIDELLM_STREAMS}, ${GUIDELLM_MAX_SECONDS}s"
  echo "Data    : prompt_tokens=${GUIDELLM_PROMPT_TOKENS}, output_tokens=${GUIDELLM_OUTPUT_TOKENS}"
  echo "Models  : ${#MODELS[@]}"
  echo "Results : $RESULTS_DIR"

  local -a run_models=("${@:-${MODELS[@]}}")
  local any_failed=false
  for model in "${run_models[@]}"; do
    run_model "$model" || any_failed=true
  done

  echo ""
  echo "=== All done ==="
  ls -lh "$RESULTS_DIR"/*"${TIMESTAMP}"* 2>/dev/null || true

  [[ "$any_failed" == true ]] && exit 1
  exit 0
}

main "$@"
