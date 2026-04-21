# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: SIM117
from pathlib import Path
from typing import Optional

import numpy as np
import openvino as ov
import torch

from openvino._offline_transformations import paged_attention_transformation
from torch import nn
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as SamplerV1

import vllm_openvino.envs as envs

logger = init_logger(__name__)


def _flatten_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(_flatten_inputs(input_data))
        elif isinstance(input_data, dict):
            flatten_inputs.extend(_flatten_inputs(list(input_data.values())))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


# Removed: _modify_cache_parameters (OpenVINO 2026.0+) and _require_model_export (local IR only)


def has_op_with_type(function: ov.Model, type_name: str):
    for op in function.get_ops():
        if op.get_type_name() == type_name:
            return True
    return False


ATTENTION_ONLY = "attention_only"
HYBRID_MAMBA = "hybrid_mamba"


def detect_model_type(ov_model: ov.Model) -> str:
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            var_id = op.get_variable_id()
            if var_id and ("ssm" in var_id or "conv" in var_id):
                return HYBRID_MAMBA
    return ATTENTION_ONLY


def get_ssm_state_shapes(ov_model: ov.Model) -> dict[str, list]:
    ssm_shapes = []
    conv_shapes = []
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            var_id = op.get_variable_id()
            if not var_id:
                continue
            if "ssm" in var_id:
                ssm_shapes.append((op.get_partial_shape(), op.get_element_type().to_string()))
            elif "conv" in var_id:
                conv_shapes.append((op.get_partial_shape(), op.get_element_type().to_string()))
    return {"ssm": ssm_shapes, "conv": conv_shapes}


def find_llm_matmul(model: ov.Model):
    last_node = model.output(0).get_node().input_value(0).get_node()

    # in case of PA all tokens are moved to batch dimension and we have to slice / gather accordingly
    pa_based_model = has_op_with_type(model, "PagedAttentionExtension")
    slice_gather_dim = 0 if pa_based_model else 1
    last_node_type = last_node.get_type_name()
    matmul = last_node
    if last_node_type == "MatMul":
        # Matmul -> Result
        return matmul, slice_gather_dim
    elif last_node_type == "Add":
        # Matmul -> Add -> Result
        matmul = last_node.input_value(0).node
    elif last_node_type == "Transpose":
        # Matmul -> Transpose -> Result
        matmul = last_node.input_value(0).node
        order = last_node.input_value(1).node.data
        slice_gather_dim = order[slice_gather_dim]
    elif last_node_type == "Multiply":
        # MatMul -> Divide -> Tanh -> Multiply -> Result
        multiply = last_node
        tanh = multiply.input_value(0).node
        if tanh.get_type_name() == "Tanh":
            divide = tanh.input_value(0).node
            if divide.get_type_name() == "Divide":
                matmul = divide.input_value(0).node
    if matmul.get_type_name() != "MatMul":
        raise ValueError(
            f"Could not find MatMul in model output. "
            f"Last node type: '{last_node_type}'. "
            "Supported output patterns: MatMul, Add->MatMul, "
            "Transpose->MatMul, Multiply->Tanh->Divide->MatMul."
        )
    return matmul, slice_gather_dim


def apply_gather_before_matmul_transformation(model: ov.Model):
    matmul, slice_gather_dim = find_llm_matmul(model)
    if matmul.get_type_name() == "MatMul" and matmul.input(0).get_partial_shape().rank == 3:
        indices = ov.op.Parameter(ov.Type.i64, ov.PartialShape([-1]))
        indices.set_friendly_name("sampled_tokens_indices")
        indices.output(0).get_tensor().set_names({"sampled_tokens_indices"})
        axis = ov.op.Constant(ov.Type.i64, ov.Shape([1]), [slice_gather_dim])
        gather = ov.opset8.gather(matmul.input_value(0), indices, axis)
        matmul.input(0).replace_source_output(gather.output(0))
        model.add_parameters([indices])


class OpenVINOCausalLM(nn.Module):

    def __init__(
        self,
        ov_core: ov.Core,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()
        self.logits_processor = LogitsProcessor(
            model_config.get_vocab_size(), logits_as_input=True)
        self.sampler = SamplerV1()

        # Only support local pre-exported IR files
        model_dir = Path(model_config.model)
        if not model_dir.is_dir():
            raise ValueError(
                f"Model path {model_config.model} is not a local directory. "
                "This plugin only supports pre-exported OpenVINO IR files. "
                "Please provide a local path containing openvino_model.xml "
                "and openvino_model.bin files.")

        # Find and load IR files
        self.use_text_embeddings_model = False
        if (model_dir / "openvino_language_model.xml").exists():
            ir_filename = "openvino_language_model.xml"
            text_emb_path = model_dir / "openvino_text_embeddings_model.xml"
            if text_emb_path.exists():
                self.use_text_embeddings_model = True
        else:
            ir_filename = "openvino_model.xml"

        ov_model = ov_core.read_model(str(model_dir / ir_filename))

        # Detect model type before PA transformation
        self.model_type = detect_model_type(ov_model)
        self.ssm_state_shapes = get_ssm_state_shapes(ov_model)

        # apply Paged Attention transformation
        paged_attention_transformation(ov_model)
        apply_gather_before_matmul_transformation(ov_model)
        # OpenVINO version guard removed: 2026.0+ no longer requires manual KV cache patching
        ov_model.validate_nodes_and_infer_types()

        ov_device = envs.VLLM_OPENVINO_DEVICE
        ov_compiled = ov_core.compile_model(ov_model, ov_device)
        self.ov_request = ov_compiled.create_infer_request()

        # Load text embeddings model for multimodal OV models (e.g. Gemma 3)
        if self.use_text_embeddings_model:
            text_emb_model = ov_core.read_model(
                str(model_dir / "openvino_text_embeddings_model.xml"))
            ov_text_emb_compiled = ov_core.compile_model(
                text_emb_model, ov_device)
            self.text_emb_request = ov_text_emb_compiled.create_infer_request()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[ov.Tensor, ov.Tensor]],
    ) -> torch.Tensor:
        flat_kv_caches = _flatten_inputs(kv_caches)
        attn_metadata = get_forward_context().attn_metadata

        if self.use_text_embeddings_model:
            # Gemma 3 style: language model takes inputs_embeds, not input_ids.
            # Run text embeddings model first: input_ids (1D) → inputs_embeds.
            input_ids_np = np.array(input_ids.data).reshape(1, -1)
            self.text_emb_request.infer([input_ids_np])
            inputs_embeds = self.text_emb_request.get_output_tensor(0)
            # inputs_embeds shape: [1, seq_len, hidden] → flatten to [seq_len, hidden]
            inputs_embeds_2d = inputs_embeds.data.reshape(-1, inputs_embeds.shape[-1])
            token_type_ids = np.zeros(
                (1, input_ids_np.shape[1]), dtype=np.int64)
            inputs = [
                positions,
                token_type_ids,
                inputs_embeds_2d,
                *flat_kv_caches,
                attn_metadata.past_lens,
                attn_metadata.subsequence_begins,
                attn_metadata.block_indices,
                attn_metadata.block_indices_begins,
                attn_metadata.max_context_len,
            ]
        else:
            inputs = [
                input_ids,
                positions,
                *flat_kv_caches,
                attn_metadata.past_lens,
                attn_metadata.subsequence_begins,
                attn_metadata.block_indices,
                attn_metadata.block_indices_begins,
                attn_metadata.max_context_len,
            ]

        inputs.append(attn_metadata.sampled_token_indices)

        self.ov_request.start_async(inputs, share_inputs=True)
        self.ov_request.wait()

        logits = torch.from_numpy(self.ov_request.get_tensor("logits").data)

        # NOTE: view reshapes logits from [seq_len, vocab] to [-1, vocab].
        # OpenVINO PA currently outputs with a seq_len dimension; remove view if/when that changes.
        return logits.view(-1, logits.shape[-1])

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:

        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits


    @torch._dynamo.disable
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


def get_model(
    vllm_config: VllmConfig,
    ov_core: ov.Core,
) -> torch.nn.Module:
    with set_current_vllm_config(vllm_config):
        return OpenVINOCausalLM(ov_core, vllm_config.model_config)

 
