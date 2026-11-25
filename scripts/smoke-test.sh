#!/usr/bin/env bash
# smoke-test.sh — Regression smoke test
# Checks the pure runner logic and the vLLM symbols the plugin imports, then
# runs every serving path (PagedAttention, Hybrid-PA) with 5 fixed greedy
# questions and a concurrent burst, tracking tok/s and a hash of the output.
# Compares against ./scripts/smoke-test-baseline.json if present: a tok/s drop
# fails, a changed output hash is reported for review (expected after an
# OpenVINO, vLLM or IR change; otherwise a correctness regression).

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

API_URL="http://127.0.0.1:8080"
CONTAINER_NAME="vllm-smoke"
IMAGE="${IMAGE:-quay.io/joopark/vllm-openvino}"
# Extra ENV=VALUE pairs for every container, e.g. SMOKE_ENV=VLLM_USE_RUST_FRONTEND=1
SMOKE_ENV_ARGS=()
for opt in ${SMOKE_ENV:-}; do SMOKE_ENV_ARGS+=(-e "$opt"); done
MODEL_BASE="${MODEL_BASE:-$HOME/hf/OpenVINO}"

# "<model dir> [ENV=VALUE | --serve-arg]..."
MODELS=(
  "Qwen3-1.7B-int4-ov"
  "gemma-4-E2B-it-int4-ov"
  "LFM3-1.7B-int4-ov"
  "Qwen3.5-0.8B-int4-ov"
  # Hybrid-PA without block checkpoints: per-request state slots, draft slots.
  "Qwen3.5-0.8B-int4-ov --no-enable-prefix-caching --speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":2}"
  "Qwen2.5-Coder-0.5B-Instruct-int4-ov"
)
CONCURRENT_REQUESTS=4

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

# Waits up to 5 minutes for the API, giving up as soon as the server exits.
wait_ready() {
  for _ in $(seq 1 60); do
    curl -sf "$API_URL/v1/models" >/dev/null 2>&1 && return 0
    [[ "$(podman inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]] || break
    sleep 5
  done
  echo "  FAIL: server did not start"
  podman logs "$CONTAINER_NAME" 2>&1 | grep -v "^\s*\^*$" | grep -B2 -A8 "Error" | tail -24
  cleanup_container
  return 1
}

run_model_perf() {
  local model_name opts
  read -r model_name opts <<< "$1"
  local label="$model_name${opts:+ ($opts)}"
  local env_args=() serve_args=() opt
  for opt in $opts; do
    if [[ "$opt" == --* ]]; then
      serve_args+=("$opt")
    else
      env_args+=(-e "$opt")
    fi
  done
  local model_dir="$MODEL_BASE/$model_name"

  if [[ ! -d "$model_dir" ]]; then
    echo "  SKIP: model dir not found: $model_dir"
    return 1
  fi

  echo ""
  echo "=== $label ==="
  cleanup_container

  podman run --replace -d --name "$CONTAINER_NAME" \
    -p 8080:8080 --cpus=8 --memory=16g \
    -v "$PROJECT_ROOT/vllm_openvino:/opt/app-root/vllm_openvino:Z" \
    -v "$model_dir:/models:Z" \
    "${SMOKE_ENV_ARGS[@]}" \
    "${env_args[@]}" \
    "$IMAGE" \
    --port=8080 --model /models --max-model-len 4096 "${serve_args[@]}" >/dev/null

  wait_ready || return 1

  local total_ms=0
  local total_tokens=0
  local q_idx=0
  local outputs=""

  for q in "${QUESTIONS[@]}"; do
    q_idx=$((q_idx + 1))
    local start
    start=$(date +%s%N)
    local resp
    resp=$(curl -sf "$API_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"/models\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0}" 2>/dev/null)
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

    outputs+=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'])" 2>/dev/null)
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

  local codes
  codes=$(for _ in $(seq 1 "$CONCURRENT_REQUESTS"); do
    curl -s -o /dev/null -w "%{http_code}\n" "$API_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"/models","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' &
  done; wait)
  if [[ $(grep -c '^200$' <<< "$codes") -ne $CONCURRENT_REQUESTS ]]; then
    echo "  FAIL: concurrent requests returned" $codes
    cleanup_container
    return 1
  fi
  echo "  Concurrent x$CONCURRENT_REQUESTS: OK"

  local avg_tps
  avg_tps=$(python3 -c "print(f'{$total_tokens / ($total_ms/1000.0):.2f}')" 2>/dev/null || echo "0.00")
  echo "  Avg: ${avg_tps} tok/s (total ${total_ms}ms, ${total_tokens} tokens)"

  local output_sha
  output_sha=$(printf '%s' "$outputs" | sha256sum | cut -c1-16)
  echo "  Output sha: $output_sha"

  cleanup_container

  python3 - "$label" "$total_ms" "$total_tokens" "$avg_tps" "$output_sha" <<'PYEOF'
import json, os, sys
model_name, total_ms, total_tokens, avg_tps, output_sha = sys.argv[1:6]
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
    "output_sha": output_sha,
}
with open(results_file, "w") as f:
    json.dump(existing, f, indent=2)
PYEOF
}

# Pure runner logic and the vLLM symbols the plugin imports, inside the image.
run_static_checks() {
  echo ""
  echo "=== Static checks ==="
  podman run --rm \
    -v "$PROJECT_ROOT/vllm_openvino:/opt/app-root/vllm_openvino:Z" \
    -v "$PROJECT_ROOT/scripts:/opt/app-root/scripts:Z" \
    --entrypoint sh "$IMAGE" -c \
    "python scripts/check-vllm-imports.py && python scripts/check-logic.py" 2>&1 \
    | grep -v "torch/_\|^(MainProcess"
  local rc=${PIPESTATUS[0]}
  (( rc == 0 )) || echo "  FAIL: static checks"
  return $rc
}

# Preempt-and-resume: a pool of about 2.4k tokens with 8 concurrent 400-token
# requests forces vLLM to preempt and later resume requests.
run_preemption_check() {
  local model_dir="$MODEL_BASE/Qwen3-1.7B-int4-ov"
  [[ -d "$model_dir" ]] || { echo "  SKIP: model dir not found: $model_dir"; return 0; }
  echo ""
  echo "=== Preemption (Qwen3-1.7B-int4-ov, VLLM_OPENVINO_KVCACHE_SPACE=0.15) ==="
  cleanup_container
  podman run --replace -d --name "$CONTAINER_NAME" \
    -p 8080:8080 --cpus=8 --memory=16g \
    -v "$PROJECT_ROOT/vllm_openvino:/opt/app-root/vllm_openvino:Z" \
    -v "$model_dir:/models:Z" \
    -e VLLM_OPENVINO_KVCACHE_SPACE=0.15 \
    "${SMOKE_ENV_ARGS[@]}" \
    "$IMAGE" --port=8080 --model /models --max-model-len 2048 >/dev/null
  wait_ready || return 1
  python3 - "$API_URL" <<'PYEOF'
import concurrent.futures as cf, json, re, sys, urllib.request
url = sys.argv[1]
def preemptions():
    text = urllib.request.urlopen(url + "/metrics").read().decode()
    m = re.search(r"^vllm:num_preemptions_total\{[^}]*\} (\S+)", text, re.M)
    return float(m.group(1)) if m else 0.0
def run(i):
    body = {"model": "/models", "prompt": f"Write a long essay, part {i}.\n\n",
            "max_tokens": 400, "temperature": 0, "ignore_eos": True}
    if i % 2:
        body["repetition_penalty"] = 1.3
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["usage"]["completion_tokens"]
before = preemptions()
with cf.ThreadPoolExecutor(8) as ex:
    tokens = list(ex.map(run, range(8)))
count = preemptions() - before
print(f"  preemptions={count:.0f} completion_tokens={tokens}")
sys.exit(0 if count > 0 and tokens == [400] * 8 else 1)
PYEOF
  local rc=$?
  if podman logs "$CONTAINER_NAME" 2>&1 | grep -q Traceback; then
    echo "  FAIL: Traceback in server log"
    rc=1
  fi
  cleanup_container
  (( rc == 0 )) && echo "  Preemption: OK" || echo "  FAIL: preemption check"
  return $rc
}

# Hybrid-PA waves next to drafts: two identical prompts in one step make the
# second hit blocks the first fills in that step (split into waves) while an
# MTP request drafts; this once crashed the engine with an IndexError.
run_spec_prefix_check() {
  local model_dir="$MODEL_BASE/Qwen3.5-0.8B-int4-ov"
  [[ -d "$model_dir" ]] || { echo "  SKIP: model dir not found: $model_dir"; return 0; }
  echo ""
  echo "=== Same-step prefix hit next to MTP drafts (Qwen3.5-0.8B-int4-ov) ==="
  cleanup_container
  podman run --replace -d --name "$CONTAINER_NAME" \
    -p 8080:8080 --cpus=8 --memory=16g \
    -v "$PROJECT_ROOT/vllm_openvino:/opt/app-root/vllm_openvino:Z" \
    -v "$model_dir:/models:Z" \
    "${SMOKE_ENV_ARGS[@]}" \
    "$IMAGE" --port=8080 --model /models --max-model-len 4096 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' >/dev/null
  wait_ready || return 1
  python3 - "$API_URL" <<'PYEOF'
import concurrent.futures as cf, json, re, sys, time, urllib.request
url = sys.argv[1]
def post(body):
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["usage"]["completion_tokens"]
def metric(name):
    text = urllib.request.urlopen(url + "/metrics").read().decode()
    m = re.search(rf"^{name}\{{[^}}]*\}} (\S+)", text, re.M)
    return float(m.group(1)) if m else 0.0
results = []
try:
    with cf.ThreadPoolExecutor(3) as ex:
        for rnd in range(3):
            drafting = ex.submit(post, {"model": "/models", "prompt": f"Count from 1 to 500, round {rnd}: 1, 2, 3,",
                                        "max_tokens": 300, "temperature": 0, "ignore_eos": True})
            time.sleep(2)
            shared = f"Round {rnd}. " + " ".join(f"item{i}" for i in range(400)) + "\nSummarize:"
            pair = [ex.submit(post, {"model": "/models", "prompt": shared, "max_tokens": 16,
                                     "temperature": 0, "ignore_eos": True}) for _ in range(2)]
            results += [f.result() for f in pair] + [drafting.result()]
except Exception as e:
    print(f"  request failed: {e}")
    sys.exit(1)
drafts = metric("vllm:spec_decode_num_draft_tokens_total")
hits = metric("vllm:prefix_cache_hits_total")
print(f"  completion_tokens={results} drafts={drafts:.0f} prefix_hit_tokens={hits:.0f}")
sys.exit(0 if results == [16, 16, 300] * 3 and drafts > 0 and hits > 0 else 1)
PYEOF
  local rc=$?
  if podman logs "$CONTAINER_NAME" 2>&1 | grep -q Traceback; then
    echo "  FAIL: Traceback in server log"
    rc=1
  fi
  cleanup_container
  (( rc == 0 )) && echo "  Prefix hit next to drafts: OK" || echo "  FAIL: prefix hit next to drafts"
  return $rc
}

main() {
  echo "=== vllm-openvino Performance Smoke Test ==="
  printf 'Models:'; printf ' [%s]' "${MODELS[@]}"; echo
  echo "Questions: ${#QUESTIONS[@]} | max_tokens=$MAX_TOKENS"

  GIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
  echo "{}" > "$RESULTS_FILE"
  export RESULTS_FILE GIT_SHA

  local any_failed=false
  run_static_checks || any_failed=true
  for model in "${MODELS[@]}"; do
    if ! run_model_perf "$model"; then
      any_failed=true
    fi
  done
  run_preemption_check || any_failed=true
  run_spec_prefix_check || any_failed=true

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
changed = []
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
    base_out = base[model].get('output_sha')
    if base_out and base_out != new_data.get('output_sha'):
        status += ', OUTPUT CHANGED'
        changed.append(model)
    print(f'  {model}: {base_tps:.2f} -> {new_tps:.2f} tok/s ({pct:+.1f}%) [{status}]')
if changed:
    print(f'REVIEW: greedy output changed for {len(changed)} model(s); expected '
          'only after an OpenVINO, vLLM or IR change.')
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
