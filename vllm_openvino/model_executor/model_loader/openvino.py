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
                ssm_shapes.append((op.output(0).get_partial_shape(), op.get_element_type().to_string()))
            elif "conv" in var_id:
                conv_shapes.append((op.output(0).get_partial_shape(), op.get_element_type().to_string()))
    return {"ssm": ssm_shapes, "conv": conv_shapes}


def _is_attention_sdpa_var(var_id: str) -> bool:
    """Check if variable_id is for attention KV cache (not SSM/conv).

    Attention KV: cache_params.past.key.{N}, cache_params.past.value.{N}
    SSM/conv: cache_params.past.ssm.{N}, cache_params.past.conv.{N}
    """
    if not var_id:
        return False
    if "past.ssm." in var_id or "past.conv." in var_id:
        return False
    if "past.key." in var_id or "past.value." in var_id:
        return True
    return False


def _find_sdpa_consumer_of_var(model: ov.Model, var_id: str) -> list:
    """Find all SDPA ops that consume a given variable_id via ReadValue."""
    consumers = []
    for op in model.get_ops():
        if op.get_type_name() != "ScaledDotProductAttention":
            continue
        for i in range(op.get_input_size()):
            try:
                source = op.input_value(i).get_node()
                if source.get_type_name() == "ReadValue" and source.get_variable_id() == var_id:
                    consumers.append(op)
                    break
            except Exception:
                continue
    return consumers


def _remove_ssmlike_sdpa_subgraph(model: ov.Model) -> None:
    """Remove SDPA ops that are part of SSM/conv (linear attention) layers.

    These SDPA ops are NOT true attention - they are part of Mamba/SSM blocks.
    We identify them by checking if their ReadValue inputs use ssm/conv variable_ids.
    """
    # Find all ssm/conv variable_ids
    ssm_conv_vars = set()
    for op in model.get_ops():
        if op.get_type_name() != "ReadValue":
            continue
        var_id = op.get_variable_id()
        if not var_id:
            continue
        if not _is_attention_sdpa_var(var_id) and ("ssm" in var_id or "conv" in var_id):
            ssm_conv_vars.add(var_id)

    # Find SDPA ops that consume ssm/conv variables
    ssm_sdpa_ops = set()
    for var_id in ssm_conv_vars:
        for op in _find_sdpa_consumer_of_var(model, var_id):
            ssm_sdpa_ops.add(op)

    # Bypass SSM/conv SDPA ops before PA transformation
    for sdpa_op in ssm_sdpa_ops:
        try:
            sdpa_output = sdpa_op.output(0)
            sdpa_input = sdpa_op.input_value(0)
            sdpa_output.replace_source_output(sdpa_input)
        except Exception:
            # Best effort: keep node unchanged if rewrite fails
            pass


def _has_sdpa_ops(model: ov.Model) -> bool:
    """Check if model has ScaledDotProductAttention operations."""
    for op in model.get_ops():
        if op.get_type_name() == "ScaledDotProductAttention":
            return True
    return False


def apply_selective_paged_attention_transformation(model: ov.Model, model_type: str) -> None:
    """Apply PA transformation selectively based on model type.

    For HYBRID_MAMBA models: PA transformation is skipped due to PrevSequenceLengthPattern
    crash on SSM Gather/Reshape nodes. The model runs with internal KV cache.
    For ATTENTION_ONLY models: apply PA transformation only if SDPA ops exist.
    """
    if model_type == ATTENTION_ONLY:
        if not _has_sdpa_ops(model):
            logger.warning(
                "Model does not have ScaledDotProductAttention operations. "
                "Skipping PagedAttention transformation. "
                "The model will run with internal KV cache."
            )
            return
        paged_attention_transformation(model)
        return

    # For hybrid models: skip PA transformation entirely due to C++ pattern matcher
    # crashing on SSM subgraphs (PrevSequenceLengthPattern on Gather/Reshape nodes).
    # The model will use internal KV cache mechanism instead of PagedAttention.
    logger.warning(
        "Hybrid Mamba models do not support PagedAttention transformation yet. "
        "The model will run with internal KV cache."
    )


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
        self.use_vision_embeddings_model = False
        if (model_dir / "openvino_language_model.xml").exists():
            ir_filename = "openvino_language_model.xml"
            text_emb_path = model_dir / "openvino_text_embeddings_model.xml"
            if text_emb_path.exists():
                self.use_text_embeddings_model = True
            vision_emb_path = model_dir / "openvino_vision_embeddings_model.xml"
            if vision_emb_path.exists():
                self.use_vision_embeddings_model = True
        else:
            ir_filename = "openvino_model.xml"

        ov_model = ov_core.read_model(str(model_dir / ir_filename))

        # Detect model type before PA transformation
        self.model_type = detect_model_type(ov_model)
        self.ssm_state_shapes = get_ssm_state_shapes(ov_model)

        # apply Paged Attention transformation (selective for hybrid models)
        apply_selective_paged_attention_transformation(ov_model, self.model_type)
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

        if self.use_vision_embeddings_model:
            vision_emb_model = ov_core.read_model(
                str(model_dir / "openvino_vision_embeddings_model.xml"))
            ov_vision_emb_compiled = ov_core.compile_model(
                vision_emb_model, ov_device)
            self.vision_emb_request = ov_vision_emb_compiled.create_infer_request()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[ov.Tensor, ov.Tensor]],
        ssm_caches: Optional[list[ov.Tensor]] = None,
        conv_caches: Optional[list[ov.Tensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_kv_caches = _flatten_inputs(kv_caches)
        state_tensors = list(flat_kv_caches)
        if self.model_type == HYBRID_MAMBA:
            state_tensors.extend(_flatten_inputs(ssm_caches or []))
            state_tensors.extend(_flatten_inputs(conv_caches or []))

        attn_metadata = get_forward_context().attn_metadata
        block_indices_groups = getattr(attn_metadata, "block_indices_groups", None)
        block_indices_begins_groups = getattr(
            attn_metadata, "block_indices_begins_groups", None)

        if block_indices_groups and block_indices_begins_groups and \
                len(block_indices_groups) == len(block_indices_begins_groups):
            block_table_inputs = []
            for block_indices, block_indices_begins in zip(
                    block_indices_groups, block_indices_begins_groups):
                block_table_inputs.extend([block_indices, block_indices_begins])
        else:
            block_table_inputs = [
                attn_metadata.block_indices,
                attn_metadata.block_indices_begins,
            ]

        if self.use_text_embeddings_model:
            # Gemma 3 style: language model takes inputs_embeds, not input_ids.
            # Run text embeddings model first: input_ids (1D) → inputs_embeds.
            input_ids_np = np.array(input_ids.data).reshape(1, -1)
            self.text_emb_request.infer([input_ids_np])
            inputs_embeds = self.text_emb_request.get_output_tensor(0)
            # inputs_embeds shape: [1, seq_len, hidden] → flatten to [seq_len, hidden]
            inputs_embeds_2d = inputs_embeds.data.reshape(-1, inputs_embeds.shape[-1])

            if self.use_vision_embeddings_model and pixel_values is not None:
                pixel_values_np = np.array(pixel_values.data)
                image_pos_np = np.array(image_position_ids.data)
                self.vision_emb_request.infer([pixel_values_np, image_pos_np])
                vision_embeds = self.vision_emb_request.get_output_tensor(0)
                vision_embeds_2d = vision_embeds.data.reshape(
                    -1, vision_embeds.shape[-1])

                for i in range(image_pos_np.shape[1]):
                    start, end = image_pos_np[0, i]
                    num_patches = end - start
                    patch_offset = i * num_patches
                    inputs_embeds_2d[start:end] = vision_embeds_2d[patch_offset:patch_offset+num_patches]

            token_type_ids = np.zeros(
                (1, input_ids_np.shape[1]), dtype=np.int64)
            inputs = [
                positions,
                token_type_ids,
                inputs_embeds_2d,
                *state_tensors,
                attn_metadata.past_lens,
                attn_metadata.subsequence_begins,
                *block_table_inputs,
                attn_metadata.max_context_len,
            ]
        else:
            inputs = [
                input_ids,
                positions,
                *state_tensors,
                attn_metadata.past_lens,
                attn_metadata.subsequence_begins,
                *block_table_inputs,
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

 
