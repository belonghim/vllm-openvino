# Podman Testing

Use a source mount to test Python changes without rebuilding the image. Keep
`--cpus=8 --memory=16g` for comparable measurements.

## Start a Server

```bash
podman run --replace -d --name vllm-server -p 8080:8080 --cpus=8 --memory=16g \
  -v ~/prj/vllm-openvino/vllm_openvino:/opt/app-root/vllm_openvino:Z \
  -v ~/hf:/models:Z \
  quay.io/joopark/vllm-openvino \
  --port=8080 --model <model_dir> --max-model-len 4096
```

Wait for startup and inspect errors:

```bash
podman logs -f vllm-server
podman logs vllm-server 2>&1 | grep -A10 'ERROR\|Traceback'
```

The source mount replaces only `vllm_openvino`; vLLM and OpenVINO come from
the image. Model paths are mounted from host `~/hf` to container `/models`.

## Validate Requests

Run the syntax check before starting the container:

```bash
python3 -m py_compile <changed_file.py>
```

Check readiness and issue API requests against the container:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model_dir>","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

For runtime changes, verify a single request, repeated sequential requests,
and concurrent requests. For multimodal changes, include an image request and
allow substantially more time for CPU vision inference.

Stop the test container when finished:

```bash
podman rm -f vllm-server
```
