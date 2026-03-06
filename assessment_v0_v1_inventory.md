# V0 vs V1 Components Inventory

This document inventories components related to V0 and V1 versions within the vLLM OpenVINO integration.

## V0/V1 Selection Logic

The selection between V0 and V1 components is determined by the `VLLM_USE_V1` environment variable, as defined in `vllm_openvino/platform.py` (lines 77-83):

- If `VLLM_USE_V1` is true, `vllm_openvino.worker_v1.openvino_worker_v1.OpenVINOWorkerV1` is used.
- If `VLLM_USE_V1` is false, `vllm_openvino.worker.openvino_worker.OpenVINOWorker` is used.

## Component Files

The following files have been identified as relevant to V0 and V1 components:

-   **V0 Worker:** `/home/user/project/vllm-openvino/vllm_openvino/worker/openvino_worker.py`
    -   Contains `OpenVINOWorker` and `OpenVINOCacheEngine`.
-   **V1 Worker:** `/home/user/project/vllm-openvino/vllm_openvino/worker_v1/openvino_worker_v1.py`
    -   Contains `OpenVINOWorkerV1`.
-   **V0 Model Runner:** `/home/user/project/vllm-openvino/vllm_openvino/worker/openvino_model_runner.py`
    -   This file exists and is likely associated with V0.
-   **V1 Model Runner:** `/home/user/project/vllm-openvino/vllm_openvino/worker_v1/openvino_model_runner_v1.py`
    -   This file exists and is likely associated with V1.
-   **Attention Backend:** `/home/user/project/vllm-openvino/vllm_openvino/attention/backends/openvino.py`
    -   This file exists and appears to be shared or V1-compatible.

## V0 Symbols Imported by V1 Code

-   **Imported Symbol**: `OpenVINOCacheEngine`
-   **Source File (V1)**: `vllm_openvino/worker_v1/openvino_worker_v1.py`
-   **Imported From**: `vllm_openvino.worker.openvino_worker`

This indicates a direct dependency of the V1 worker on a V0-specific component (`OpenVINOCacheEngine`). This component will need to be refactored or reimplemented within the V1 structure to remove the V0 dependency.

## Refactoring/Removal Considerations

The primary dependency of V1 code on V0 code is the import of `OpenVINOCacheEngine` by `OpenVINOWorkerV1`. This suggests that `OpenVINOCacheEngine` should either be:
1. Refactored to be V1-compatible and moved to a shared location.
2. Re-implemented within the V1 worker or a new V1-specific cache module if its functionality is significantly different or tied to V0-specific patterns.
3. The `OpenVINOWorkerV1` should be updated to use a V1-native cache mechanism if one exists or is to be created.

The `OpenVINOModelRunner` (V0) and `OpenVINOModelRunnerV1` (V1) appear to be distinct implementations, which is good. However, the `OpenVINOAttentionMetadata` is shared, implying that the underlying attention mechanisms might be compatible or that the V1 runner is adapted to use the V0 metadata structure. Further investigation into `OpenVINOAttentionMetadata` and its usage in `OpenVINOModelRunnerV1` might be needed if it relies on V0-specific assumptions.