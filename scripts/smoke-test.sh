#!/usr/bin/env bash
# smoke-test.sh — Performance regression smoke test
# Runs PA(Qwen) + Stateful(gemma-4) with 5 fixed questions, tracks tok/s.
# Compares against ./scripts/smoke-test-baseline.json if present.

set -uo pipefail

UPDATE_BASELINE=false
for arg in "$@"; do
  [[ "$arg" == "--update-baseline" ]] && UPDATE_BASELINE=true
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_FILE="$SCRIPT_DIR/smoke-test-results.json"
BASELINE_FILE="$SCRIPT_DIR/smoke-test-baseline.json"
REGRESSION_THRESHOLD_PCT=10

API_URL="http://localhost:8080"
CONTAINER_NAME="vllm-smoke"
IMAGE="quay.io/joopark/vllm-openvino"
MODEL_BASE="/home/user/hf/OpenVINO"

MODELS=(
  "Qwen2.5-Coder-3B-Instruct-int4-ov"
  "gemma-4-E4B-it-int4-ov"
)

QUESTIONS=(
  "Write a Python function that checks if a string is a palindrome."
  "What is the time complexity of quicksort in the average case and why?"
  "Explain the difference between a process and a thread in one paragraph."
  "Write a SQL query to find the top 3 customers by total order amount."
  "What is the difference between TCP and UDP? Give a concrete example of when to use each."
)

MAX_TOKENS=256

cleanup_container() {
  podman stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  podman rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

run_model_perf() {
  local model_name="$1"
  local model_dir="$MODEL_BASE/$model_name"

  if [[ ! -d "$model_dir" ]]; then
    echo "  SKIP: model dir not found: $model_dir"
    return 1
  fi

  echo ""
  echo "=== $model_name ==="
  cleanup_container

  podman run --replace -d --name "$CONTAINER_NAME" \
    -p 8080:8080 \
    -v "$PROJECT_ROOT/vllm_openvino:/opt/app-root/vllm_openvino" \
    -v "$model_dir:/models:Z" \
    -e VLLM_OPENVINO_DEVICE=CPU \
    -e TORCH_COMPILE_DISABLE=1 \
    "$IMAGE" \
    --port=8080 --model /models --max-model-len 4096 >/dev/null

  local ready=false
  for _ in $(seq 1 60); do
    if curl -sf "$API_URL/v1/models" >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 5
  done

  if [[ "$ready" != true ]]; then
    echo "  FAIL: container startup timeout"
    podman logs "$CONTAINER_NAME" 2>&1 | tail -10
    cleanup_container
    return 1
  fi

  local total_ms=0
  local total_tokens=0
  local q_idx=0

  for q in "${QUESTIONS[@]}"; do
    q_idx=$((q_idx + 1))
    local start
    start=$(date +%s%N)
    local resp
    resp=$(curl -sf "$API_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"/models\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":$MAX_TOKENS}" 2>/dev/null)
    local end
    end=$(date +%s%N)
    local ms=$(( (end - start) / 1000000 ))

    if [[ -z "$resp" ]]; then
      echo "  Q$q_idx FAIL: no response"
      cleanup_container
      return 1
    fi

    local tokens
    tokens=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo 0)

    total_ms=$((total_ms + ms))
    total_tokens=$((total_tokens + tokens))

    local tps
    tps=$(python3 -c "print(f'{$tokens / ($ms/1000.0):.1f}')" 2>/dev/null || echo "0.0")

    local content
    content=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d['choices'][0]['message']['content'] if d.get('choices') else ''
if '</think>' in text:
    text = text.split('</think>')[-1]
text = text.strip().replace('\n', ' ')
print(text[:160] + ('...' if len(text) > 160 else ''))
" 2>/dev/null || echo "")

    printf "  Q%d: %6dms / %3d tok / %s tok/s  [%s]\n" "$q_idx" "$ms" "$tokens" "$tps" "${q:0:45}..."
    [[ -n "$content" ]] && echo "       $content"
  done

  local avg_tps
  avg_tps=$(python3 -c "print(f'{$total_tokens / ($total_ms/1000.0):.2f}')" 2>/dev/null || echo "0.00")
  echo "  Avg: ${avg_tps} tok/s (total ${total_ms}ms, ${total_tokens} tokens)"

  cleanup_container

  python3 - "$model_name" "$total_ms" "$total_tokens" "$avg_tps" <<'PYEOF'
import json, os, sys
model_name, total_ms, total_tokens, avg_tps = sys.argv[1:5]
results_file = os.environ["RESULTS_FILE"]
git_sha = os.environ.get("GIT_SHA", "unknown")
existing = {}
if os.path.exists(results_file):
    try:
        with open(results_file) as f:
            existing = json.load(f)
    except Exception:
        existing = {}
existing["_meta"] = {"git_sha": git_sha}
existing[model_name] = {
    "total_ms": int(total_ms),
    "total_tokens": int(total_tokens),
    "avg_tps": float(avg_tps),
}
with open(results_file, "w") as f:
    json.dump(existing, f, indent=2)
PYEOF
}

main() {
  echo "=== vllm-openvino Performance Smoke Test ==="
  echo "Models: ${MODELS[*]}"
  echo "Questions: ${#QUESTIONS[@]} | max_tokens=$MAX_TOKENS"

  GIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
  echo "{}" > "$RESULTS_FILE"
  export RESULTS_FILE GIT_SHA

  local any_failed=false
  for model in "${MODELS[@]}"; do
    if ! run_model_perf "$model"; then
      any_failed=true
    fi
  done

  echo ""
  echo "Results saved: $RESULTS_FILE"

  if [[ "$UPDATE_BASELINE" == "true" ]]; then
    cp "$RESULTS_FILE" "$BASELINE_FILE"
    echo "Baseline updated: $BASELINE_FILE"
    [[ "$any_failed" == true ]] && exit 1
    exit 0
  fi

  if [[ -f "$BASELINE_FILE" ]]; then
    echo ""
    echo "=== Regression Check (threshold: ${REGRESSION_THRESHOLD_PCT}%) ==="
    python3 - "$RESULTS_FILE" "$BASELINE_FILE" "$REGRESSION_THRESHOLD_PCT" <<'PYEOF'
import json, sys
results_file, baseline_file, threshold_pct = sys.argv[1:4]
threshold = float(threshold_pct)
with open(results_file) as f:
    new = json.load(f)
with open(baseline_file) as f:
    base = json.load(f)
base_sha = base.get('_meta', {}).get('git_sha', 'unknown')
new_sha = new.get('_meta', {}).get('git_sha', 'unknown')
print(f'  Baseline: {base_sha} -> Current: {new_sha}')
regressed = False
for model, new_data in new.items():
    if model.startswith('_'):
        continue
    if model not in base:
        print(f'  {model}: NEW (no baseline)')
        continue
    base_tps = base[model].get('avg_tps', 0.0)
    new_tps = new_data.get('avg_tps', 0.0)
    if base_tps <= 0:
        continue
    pct = (new_tps - base_tps) / base_tps * 100
    status = 'OK'
    if pct < -threshold:
        status = 'REGRESSION'
        regressed = True
    elif pct > threshold:
        status = 'IMPROVED'
    print(f'  {model}: {base_tps:.2f} -> {new_tps:.2f} tok/s ({pct:+.1f}%) [{status}]')
sys.exit(1 if regressed else 0)
PYEOF
    local rc=$?
    if (( rc != 0 )); then
      echo "FAIL: regression detected"
      exit 1
    fi
    echo "OK: no regression"
  else
    echo ""
    echo "No baseline at $BASELINE_FILE"
    echo "To save current results as baseline:"
    echo "  cp $RESULTS_FILE $BASELINE_FILE"
  fi

  if [[ "$any_failed" == true ]]; then
    exit 1
  fi
}

main "$@"
