# SPDX-License-Identifier: Apache-2.0
"""Model-related data structures for OpenVINO backend."""

from typing import List, NamedTuple, Optional
import torch
from vllm_openvino.attention.backends.openvino import OpenVINOAttentionMetadata
from vllm.multimodal import BatchedTensorInputs


class ModelInput(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: Optional[OpenVINOAttentionMetadata]
    seq_lens: List[int]
    query_lens: List[int]
    multi_modal_kwargs: BatchedTensorInputs

    @classmethod
    def empty(cls, device):
        return ModelInput(
            input_tokens=torch.empty(0, device=device),
            input_positions=torch.empty(0, device=device),
            attn_metadata=None,
            seq_lens=[],
            query_lens=[],
            multi_modal_kwargs={})
