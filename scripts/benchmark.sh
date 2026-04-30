#!/usr/bin/env bash
# benchmark.sh — Simple throughput/latency benchmark for vllm-openvino
# Usage: ./scripts/benchmark.sh <model_path> [num_requests=100]

set -euo pipefail

MODEL_PATH="${1:-}"
NUM_REQUESTS="${2:-100}"
API_URL="${API_URL:-http://localhost:8080}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 <model_path> [num_requests]"
    echo "Example: $0 /models/TinyLlama-1.1B 100"
    exit 1
fi

PROMPT="Hello, how are you doing today? I hope you're having a wonderful day so far."
MAX_TOKENS=128

echo "=== vllm-openvino Benchmark ==="
echo "Model:     $MODEL_PATH"
echo "Requests:  $NUM_REQUESTS"
echo "URL:       $API_URL"
echo ""

# Warm-up request (excluded from timing)
echo "Warming up..."
curl -sf "$API_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"$PROMPT\",\"max_tokens\":$MAX_TOKENS}" > /dev/null

echo "Benchmarking..."
START=$(date +%s.%N)

for i in $(seq 1 "$NUM_REQUESTS"); do
    curl -sf "$API_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"$PROMPT\",\"max_tokens\":$MAX_TOKENS}" > /dev/null &
    
    # Limit concurrency to avoid overwhelming the server
    if (( i % 10 == 0 )); then
        wait
    fi
done
wait

END=$(date +%s.%N)
DURATION=$(echo "$END - $START" | bc)
TOTAL_TOKENS=$((NUM_REQUESTS * MAX_TOKENS))
TPS=$(echo "scale=2; $TOTAL_TOKENS / $DURATION" | bc)

# Calculate average latency per request
AVG_LAT=$(echo "scale=3; $DURATION * 1000 / $NUM_REQUESTS" | bc)

echo ""
echo "=== Results ==="
printf "Duration:      %s seconds\n" "$DURATION"
printf "Total tokens:  %d\n" "$TOTAL_TOKENS"
printf "Throughput:    %s tokens/sec\n" "$TPS"
printf "Avg latency:   %s ms/request\n" "$AVG_LAT"
