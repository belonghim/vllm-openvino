# V0/V1 Inventory Assessment

This document inventories files related to V0 and V1 OpenVINO workers and identifies V0-specific code that needs refactoring or removal.

## File Categorization

### V0-Only Files:
- `vllm_openvino/worker/openvino_worker.py`: Defines `OpenVINOWorker` and `OpenVINOCacheEngine`.
- `vllm_openvino/worker/openvino_model_runner.py`: Defines `OpenVINOModelRunner`.

### V1-Only Files:
- `vllm_openvino/worker_v1/openvino_worker_v1.py`: Defines `OpenVINOWorkerV1`.
- `vllm_openvino/worker_v1/openvino_model_runner_v1.py`: Defines `OpenVINOModelRunnerV1`.

### Shared / V1-Dependent Files:
- `vllm_openvino/platform.py`: This file contains logic to select the worker class based on the `VLLM_USE_V1` environment variable, indicating it's used in a V1 context to choose between V0 and V1 workers.
- `vllm_openvino/attention/backends/openvino.py`: Defines `OpenVINOAttentionBackend` and `OpenVINOAttentionMetadata`. These are likely used by both V0 and V1, or adapted for V1.

## V0 Symbols Imported by V1 Code

- **Imported Symbol**: `OpenVINOCacheEngine`
- **Source File (V1)**: `vllm_openvino/worker_v1/openvino_worker_v1.py`
- **Imported From**: `vllm_openvino.worker.openvino_worker`

This indicates a direct dependency of the V1 worker on a V0-specific component (`OpenVINOCacheEngine`). This component will need to be refactored or reimplemented within the V1 structure to remove the V0 dependency.

## Refactoring/Removal Considerations

The primary dependency of V1 code on V0 code is the import of `OpenVINOCacheEngine` by `OpenVINOWorkerV1`. This suggests that `OpenVINOCacheEngine` should either be:
1. Refactored to be V1-compatible and moved to a shared location.
2. Re-implemented within the V1 worker or a new V1-specific cache module if its functionality is significantly different or tied to V0-specific patterns.
3. The `OpenVINOWorkerV1` should be updated to use a V1-native cache mechanism if one exists or is to be created.

The `OpenVINOModelRunner` (V0) and `OpenVINOModelRunnerV1` (V1) appear to be distinct implementations, which is good. However, the `OpenVINOAttentionMetadata` is shared, implying that the underlying attention mechanisms might be compatible or that the V1 runner is adapted to use the V0 metadata structure. Further investigation into `OpenVINOAttentionMetadata` and its usage in `OpenVINOModelRunnerV1` might be needed if it relies on V0-specific assumptions.
