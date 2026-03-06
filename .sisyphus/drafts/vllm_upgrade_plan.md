## vLLM Upgrade Plan Draft: v0.10.2 to v0.13.0

### Goal
Upgrade `vllm` from `v0.10.2` to `v0.13.0` while maintaining "v0-like behavior" within the `vllm-openvino` project, despite the `v0` (Volley) engine's removal in `vllm v0.11.0`.

### Current Understanding

**Codebase Structure and `vllm` Integration:**
*   The `vllm-openvino` project is structured under `vllm_openvino/` with specialized modules for attention backends, model execution, and workers (`worker/`, `worker_v1/`).
*   Core components include `platform.py` (for OpenVINO configurations and worker/backend selection), `openvino_worker.py`/`openvino_worker_v1.py` (KV cache, model execution), `openvino_model_runner.py`/`openvino_model_runner_v1.py` (model loading, inference), `model_loader/openvino.py` (model conversion to OpenVINO IR), and `attention/backends/openvino.py` (OpenVINO attention).
*   Extensive imports from core `vllm` modules (`vllm.attention`, `vllm.config`, `vllm.model_executor`, `vllm.worker`, `vllm.distributed`) are present.
*   A comment in `platform.py` refers to `v0.8.1`, hinting at past compatibility or specific configurations related to earlier `vllm` versions.

**`vllm` Version Changes (v0.10.2 to v0.13.0):**
*   **Critical Breaking Change**: `vllm v0.11.0` "completes the removal of V0 engine. V0 engine code including AsyncLLMEngine, LLMEngine, MQLLMEngine, all attention backends, and related components have been removed. V1 is the only engine in the codebase now."
*   `v0.10.2` included "V0 deprecations" and "API changes."
*   `v0.13.0` uses the V1 engine.

**Test Infrastructure:**
*   A comprehensive test suite exists under `tests/` (e.g., `test_model_executor.py`, `test_vllm_engine.py`).
*   The test file naming convention (`test_*.py`) strongly suggests `pytest` is the testing framework, but explicit confirmation is needed.
*   No standard CI configuration files (`.github/workflows/*.yml`, `Jenkinsfile`, `.gitlab-ci.yml`) were found.

**Constraints:**
*   `Containerfile` and `pyproject.toml` **must not be modified**. This is a critical constraint impacting how the `vllm` dependency upgrade can be managed.

### Open Questions for User

1.  **Clarification on "v0-like behavior"**: What specific functionalities, APIs, or internal behaviors from the V0 engine does `vllm-openvino` rely on and need to preserve or re-implement using V1 components? Please provide concrete examples if possible.
2.  **`vllm` Dependency Management**: Given the constraint that `pyproject.toml` and `Containerfile` cannot be modified, how is the `vllm` dependency currently managed in this project? How can we upgrade `vllm` to `v0.13.0` under this strict constraint?
3.  **Test Framework Confirmation**: Can you confirm that `pytest` is the testing framework used in this project?
4.  **CI/CD Process**: What CI/CD system is used, and how are tests currently executed and the project built/deployed? Knowing this will help ensure the upgraded system integrates correctly.