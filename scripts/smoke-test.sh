#!/usr/bin/env bash
# smoke-test.sh — Regression smoke test for vllm-openvino
# Tests all available models with simple arithmetic prompts.
# Usage: ./scripts/smoke-test.sh [--vision]

set -uo pipefail

ENABLE_VISION=false
for arg in "$@"; do
  if [[ "$arg" == "--vision" ]]; then
    ENABLE_VISION=true
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
API_URL="http://localhost:8080"
CONTAINER_NAME="vllm-smoke"
IMAGE="quay.io/joopark/vllm-openvino"

MODEL_BASE_DIRS=(
  "/home/user/hf/OpenVINO"
  "/home/user/hf/circulus"
)

TESTS=(
  "1+1은?:2"
  "5-2의 결과는?:3"
  "3+4의 결과는?:7"
)

total_passed=0
total_failed=0

find_models() {
  for base_dir in "${MODEL_BASE_DIRS[@]}"; do
    if [[ -d "$base_dir" ]]; then
      find "$base_dir" -maxdepth 1 -mindepth 1 -type d
    fi
  done
}

cleanup_container() {
  podman stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  podman rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

run_model_test() {
  local model_dir="$1"
  local model_name
  model_name=$(basename "$model_dir")

  echo ""
  echo "=== Testing $model_name ==="

  cleanup_container

  podman run --replace -d --name "$CONTAINER_NAME" \
    -p 8080:8080 \
    -v "$PROJECT_ROOT/vllm_openvino:/opt/app-root/vllm_openvino" \
    -v "$model_dir:/models:Z" \
    -e VLLM_OPENVINO_DEVICE=CPU \
    -e TORCH_COMPILE_DISABLE=1 \
    -e VLLM_OPENVINO_KVCACHE_SPACE=8 \
    "$IMAGE" \
    --port=8080 --model /models --max-model-len 4096

  local ready=false
  for i in $(seq 1 60); do
    if podman logs "$CONTAINER_NAME" 2>&1 | grep -q "Application startup complete"; then
      ready=true
      break
    fi
    sleep 5
  done

  if [[ "$ready" != true ]]; then
    echo "  FAIL: Container startup timeout"
    podman logs "$CONTAINER_NAME" 2>&1 | tail -20
    cleanup_container
    return 1
  fi

  local model_passed=0
  local model_failed=0

  for test_case in "${TESTS[@]}"; do
    IFS=':' read -r prompt expected <<< "$test_case"

    local response
    response=$(curl -sf "$API_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"/models\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":128}" 2>/dev/null || true)

    if [[ -z "$response" ]]; then
      echo "  FAIL: '$prompt' -> No response"
      ((model_failed++))
      continue
    fi

    local raw_content
    raw_content=$(echo "$response" | python3 -c "import sys,json; j=json.load(sys.stdin); print(j['choices'][0]['message']['content'] if j.get('choices') and j['choices'][0].get('message') else '')" 2>/dev/null || true)

    local content
    content=$(echo "$raw_content" | python3 -c "
import sys
text = sys.stdin.read()
if '</think>' in text:
    text = text.split('</think>')[-1]
elif '<think>' in text:
    text = text.split('<think>')[0]
print(text.strip())
" 2>/dev/null || echo "$raw_content")

    if [[ "$content" == *"$expected"* ]]; then
      echo "  PASS: '$prompt' -> '$content' (contains '$expected')"
      ((model_passed++))
    else
      echo "  FAIL: '$prompt' -> '$raw_content' (expected '$expected')"
      ((model_failed++))
    fi
  done

  # Vision test (only if the exported vision model accepts pixel_values)
  local has_vision=false
  if [[ -f "$model_dir/openvino_vision_embeddings_model.xml" ]]; then
    has_vision=$(python3 -c "
import openvino as ov
try:
    m = ov.Core().read_model('$model_dir/openvino_vision_embeddings_model.xml')
    names = [i.get_any_name() for i in m.inputs]
    print('true' if 'pixel_values' in names else 'false')
except Exception:
    print('false')
" 2>/dev/null)
  fi

  if [[ "$ENABLE_VISION" == "true" && "$has_vision" == "true" ]]; then
    local vision_response
    vision_response=$(curl -sf --max-time 120 "$API_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"/models","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}},{"type":"text","text":"What color is this?"}]}],"max_tokens":16}' 2>/dev/null || true)

    if [[ -z "$vision_response" ]]; then
      echo "  FAIL: vision -> No response"
      ((model_failed++))
    elif echo "$vision_response" | python3 -c "import sys,json; j=json.load(sys.stdin); sys.exit(0 if j.get('choices') else 1)" 2>/dev/null; then
      local vision_content
      vision_content=$(echo "$vision_response" | python3 -c "import sys,json; j=json.load(sys.stdin); print(j['choices'][0]['message']['content'] if j.get('choices') and j['choices'][0].get('message') else '')" 2>/dev/null || true)
      if [[ -n "$vision_content" ]]; then
        echo "  PASS: vision -> '$vision_content'"
        ((model_passed++))
      else
        echo "  FAIL: vision -> Empty content"
        ((model_failed++))
      fi
    else
      echo "  FAIL: vision -> Error response"
      ((model_failed++))
    fi
  fi

  cleanup_container

  if (( model_failed == 0 )); then
    echo "  => $model_name: ALL PASSED"
    ((total_passed++))
    return 0
  elif (( model_passed >= 2 )); then
    echo "  => $model_name: PASS ($model_passed/3, $model_failed soft failures)"
    ((total_passed++))
    return 0
  else
    echo "  => $model_name: FAILED ($model_failed failures)"
    ((total_failed++))
    return 1
  fi
}

main() {
  echo "=== vllm-openvino Smoke Test ==="
  echo "Scanning for models..."

  local models=()
  while IFS= read -r model_dir; do
    models+=("$model_dir")
    echo "  Found: $(basename "$model_dir")"
  done < <(find_models)

  if (( ${#models[@]} == 0 )); then
    echo "ERROR: No models found in ${MODEL_BASE_DIRS[*]}"
    exit 1
  fi

  echo ""
  echo "Models to test: ${#models[@]}"

  for model_dir in "${models[@]}"; do
    run_model_test "$model_dir" || true
  done

  echo ""
  echo "=== Summary ==="
  echo "Passed: $total_passed"
  echo "Failed: $total_failed"

  if (( total_failed > 0 )); then
    exit 1
  fi
}

main "$@"
