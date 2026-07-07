# Podman Source Mount Testing Guide

**빌드 없이 코드 수정을 즉시 반영**하는 방법. `vllm_openvino/` 디렉토리를 컨테이너에 마운트하면 파일 수정 후 컨테이너 재시작만으로 적용된다.

## 기본 명령어

```bash
# 서버 시작 (소스 마운트 + 모델 마운트)
podman run --replace -d --name vllm-server -p 8080:8080 \
  -v /home/user/project/vllm-openvino/vllm_openvino:/opt/app-root/vllm_openvino:Z \
  -v /home/user/hf:/models:Z \
  quay.io/joopark/vllm-openvino \
  --port=8080 --model <model_dir> --max-model-len 4096

# 서버 시작 대기 (Application startup complete 메시지 확인)
for i in $(seq 1 30); do
  if podman logs vllm-server 2>&1 | grep -q "Application startup complete"; then
    echo "Ready"; break
  fi
  if podman logs vllm-server 2>&1 | grep -q "EngineCore failed\|Engine core initialization failed"; then
    echo "FAILED"; podman logs vllm-server 2>&1 | grep -A5 "ERROR" | tail -20; break
  fi
  sleep 5
done

# 로그 확인 (에러 필터)
podman logs vllm-server 2>&1 | grep -A10 'ERROR\|Traceback' | grep -v 'Triton\|CUDA'

# 서버 중지
podman stop vllm-server

# 모델 목록 확인
curl -s http://localhost:8080/v1/models | python3 -m json.tool

# 추론 테스트 (completions)
curl -s http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model_dir>","prompt":"Hello","max_tokens":32}'

# 추론 테스트 (chat)
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model_dir>","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## 반복 디버그 루프 패턴

```
1. 코드 수정 (vllm_openvino/*.py)
2. python3 -m py_compile <파일>  ← 문법 오류 사전 차단
3. podman stop vllm-server
4. podman run --replace -d ...   ← 재시작만으로 반영 (빌드 불필요)
5. 시작 대기 → 에러 확인 → 원인 분석 → 1번으로
```

## 모델 경로 규칙

- 호스트: `~/hf/<model_dir>`
- 컨테이너: `<model_dir>`
- `-v /home/user/hf:/models:Z` 마운트로 연결 (`:Z` = SELinux 레이블)

## 주의사항

- 컨테이너 이미지 `quay.io/joopark/vllm-openvino`에 이미 vLLM + OpenVINO가 설치되어 있음
- `vllm_openvino/` 소스만 마운트로 교체 — vLLM core 등은 이미지 내 버전 사용
- 이미지 내 `vllm_openvino`는 `/opt/app-root/vllm_openvino`에 설치됨

## Vision (multimodal) 테스트

Gemma-3/4와 같은 multimodal 모델은 이미지 입력을 처리할 수 있습니다. CPU에서 vision 추론은 매우 느립니다:

- **Prefill**: 272 tokens 처리에 ~8-10초 (vision embedding 생성 포함)
- **Decode**: 토큰당 ~15-30초 (CPU 제한)
- **전체 요청**: max_tokens=16 기준 ~3-4분 소요

```bash
# Vision 요청 테스트
# 1x1 PNG 이미지 (base64) + 질문
curl -s --max-time 240 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model_dir>",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "What color is this?"}
      ]
    }],
    "max_tokens": 16
  }'
```

---

## 테스트 인프라

**테스트 인프라 없음.** pytest, conftest.py, 테스트 파일 모두 없습니다.

실제 추론 테스트에는 OpenVINO 런타임 + 모델 파일이 필요하므로, 단위 테스트의 커버리지가 제한적입니다. 에이전트는 테스트 파일 추가를 권유받았을 때만 추가하며, 자발적으로 테스트 인프라를 구축하지 마세요. (상세: `docs/decisions.md`의 "테스트 인프라 추가" 항목 참조)
