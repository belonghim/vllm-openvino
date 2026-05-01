# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: SIM117
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import openvino as ov
import torch
from statistics import StatisticsError, mode

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


class OpenVINOInputBuilder(ABC):
    """Abstract base class for building OpenVINO model inputs.

    Subclasses implement ``build_inputs()`` to prepare the input
    list or dict consumed by an OpenVINO ``InferRequest``.
    """

    @abstractmethod
    def build_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[ov.Tensor, ov.Tensor]],
        ssm_caches: list[ov.Tensor] | None = None,
        conv_caches: list[ov.Tensor] | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
    ) -> list | dict:
        """Build and return inputs for the OpenVINO inference request.

        Args:
            input_ids: Token IDs tensor.
            positions: Position IDs tensor.
            kv_caches: List of KV cache tensor pairs.
            ssm_caches: Optional SSM state tensors (hybrid models).
            conv_caches: Optional convolution cache tensors (hybrid models).
            pixel_values: Optional pixel values for vision embeddings.
            image_position_ids: Optional image position indices.

        Returns:
            A list or dict suitable for ``ov_request.infer()``.
        """
        ...


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

        apply_selective_paged_attention_transformation(ov_model, self.model_type)
        if has_op_with_type(ov_model, "PagedAttentionExtension"):
            apply_gather_before_matmul_transformation(ov_model)
        # OpenVINO version guard removed: 2026.0+ no longer requires manual KV cache patching
        ov_model.validate_nodes_and_infer_types()

        ov_device = envs.VLLM_OPENVINO_DEVICE

        perf_mode = envs.VLLM_OPENVINO_PERFORMANCE_MODE
        ov_device_upper = ov_device.upper()
        import openvino.properties.hint as hints
        import openvino.properties as props
        perf_hint = {hints.performance_mode: hints.PerformanceMode.LATENCY} \
            if perf_mode == "LATENCY" else {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}

        if ov_device_upper == "CPU":
            cpu_hint: dict[Any, Any] = {}

            cpu_threads_num = envs.VLLM_OPENVINO_CPU_THREADS_NUM
            if cpu_threads_num > 0:
                # AVX2-only CPUs are often compute-bound; capping threads can
                # reduce oversubscription and improve stable token throughput.
                cpu_hint[props.inference_num_threads] = cpu_threads_num

            cpu_bind_thread = envs.VLLM_OPENVINO_CPU_BIND_THREAD
            if cpu_bind_thread in {"CORE", "NUMA", "NONE"}:
                # Explicit affinity helps avoid thread migration penalties.
                affinity_enum = getattr(getattr(props, "Affinity", None),
                                        cpu_bind_thread, None)
                affinity_value = affinity_enum if affinity_enum is not None else cpu_bind_thread
                affinity_key = getattr(props, "affinity", "AFFINITY")
                cpu_hint[affinity_key] = affinity_value

            num_streams = envs.VLLM_OPENVINO_NUM_STREAMS
            if isinstance(num_streams, int) and num_streams > 0:
                # Multiple streams can increase throughput on CPUs by enabling
                # parallel infer request execution.
                cpu_hint[props.num_streams] = num_streams
            elif num_streams == "AUTO":
                # Keep OpenVINO stream heuristic (default behavior).
                pass

            enable_ht = envs.VLLM_OPENVINO_ENABLE_HYPER_THREADING
            if enable_ht is not None:
                cpu_hint[hints.enable_hyper_threading()] = enable_ht

            inference_prec = envs.VLLM_OPENVINO_INFERENCE_PRECISION
            if inference_prec is not None:
                cpu_hint[hints.inference_precision] = inference_prec

            enable_pinning = envs.VLLM_OPENVINO_ENABLE_CPU_PINNING
            if enable_pinning is not None:
                cpu_hint[hints.enable_cpu_pinning()] = enable_pinning

            perf_hint = {**perf_hint, **cpu_hint}

        ov_compiled = ov_core.compile_model(ov_model, ov_device, perf_hint)
        self.ov_request = ov_compiled.create_infer_request()
        self._flat_kv_caches_template: list[ov.Tensor] | None = None
        self._use_grouped_block_table_inputs: bool | None = None

        # Detect if model has external KV cache inputs (PA-transformed)
        self._has_kv_cache_inputs = any(
            inp.get_any_name().startswith(("key_cache.", "value_cache."))
            for inp in ov_compiled.inputs
        )
        logger.info(
            "OpenVINO model loaded: type=%s, has_kv_cache=%s",
            self.model_type, self._has_kv_cache_inputs
        )

        # Load text embeddings model for multimodal OV models (e.g. Gemma 3)
        if self.use_text_embeddings_model:
            text_emb_model = ov_core.read_model(
                str(model_dir / "openvino_text_embeddings_model.xml"))
            ov_text_emb_compiled = ov_core.compile_model(
                text_emb_model, ov_device, perf_hint)
            self.text_emb_request = ov_text_emb_compiled.create_infer_request()

        if self.use_vision_embeddings_model:
            vision_emb_model = ov_core.read_model(
                str(model_dir / "openvino_vision_embeddings_model.xml"))
            ov_vision_emb_compiled = ov_core.compile_model(
                vision_emb_model, ov_device, perf_hint)
            self.vision_emb_request = ov_vision_emb_compiled.create_infer_request()

    def _get_flat_kv_caches_template(
        self,
        kv_caches: list[tuple[ov.Tensor, ov.Tensor]],
    ) -> list[ov.Tensor]:
        if self._flat_kv_caches_template is None:
            self._flat_kv_caches_template = _flatten_inputs(kv_caches)
        # Return a shallow copy so per-forward state_tensors can be extended safely.
        return list(self._flat_kv_caches_template)

    @staticmethod
    def _as_numpy_no_copy(tensor_like: np.ndarray | ov.Tensor | torch.Tensor | Any) -> np.ndarray:
        if isinstance(tensor_like, np.ndarray):
            return tensor_like
        if isinstance(tensor_like, ov.Tensor):
            return tensor_like.data
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like.detach().cpu().numpy()
        assert not isinstance(tensor_like, (ov.Tensor, torch.Tensor)), \
            f"_as_numpy_no_copy: unhandled type {type(tensor_like)}"
        return np.asarray(tensor_like)

    def _iter_block_table_inputs(self, attn_metadata):
        if self._use_grouped_block_table_inputs is None:
            block_indices_groups = getattr(attn_metadata, "block_indices_groups", None)
            block_indices_begins_groups = getattr(
                attn_metadata, "block_indices_begins_groups", None)
            self._use_grouped_block_table_inputs = (
                block_indices_groups is not None
                and block_indices_begins_groups is not None
                and len(block_indices_groups) == len(block_indices_begins_groups)
            )

        if self._use_grouped_block_table_inputs:
            return zip(
                attn_metadata.block_indices_groups,
                attn_metadata.block_indices_begins_groups,
            )

        return ((attn_metadata.block_indices, attn_metadata.block_indices_begins),)

    def _get_input_builder(self) -> OpenVINOInputBuilder:
        """Factory method that returns the appropriate input builder.

        Routes based on compiled model capabilities:
        - PA-transformed models (have KV cache inputs) -> PAInputBuilder
        - Stateful models (no KV cache inputs) -> StatefulInputBuilder
        """
        if self._has_kv_cache_inputs:
            return PAInputBuilder(self)
        else:
            return StatefulInputBuilder(self)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[ov.Tensor, ov.Tensor]],
        ssm_caches: list[ov.Tensor] | None = None,
        conv_caches: list[ov.Tensor] | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        builder = self._get_input_builder()
        inputs = builder.build_inputs(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            ssm_caches=ssm_caches,
            conv_caches=conv_caches,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )
        self.ov_request.infer(inputs)
        logits = torch.from_numpy(self.ov_request.get_tensor("logits").data)

        # For stateful models, logits shape is [batch, seq_len, vocab].
        # vLLM expects [batch, vocab] (last token logits only).
        if logits.dim() == 3:
            if logits.shape[1] > 0:
                # Only return logits for the first item in batch
                # (vLLM processes one request at a time for now)
                logits = logits[0:1, -1, :]  # [1, vocab]
            else:
                logger.warning("logits seq_len is 0, returning as-is")
        elif logits.dim() == 2:
            if logits.shape[0] > 0:
                logits = logits[-1:, :]  # [1, vocab]
            else:
                logger.warning("logits batch is 0, returning as-is")
        
        return logits

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # Stateful models return logits directly because the OpenVINO IR
        # includes the final lm_head/matmul layer. PA-transformed models
        # return hidden states and need the logits_processor.
        if not self._has_kv_cache_inputs:
            return hidden_states
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits


    @torch._dynamo.disable
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


class PAInputBuilder(OpenVINOInputBuilder):
    """Builds list-based inputs for PA-transformed OpenVINO models.

    This builder encapsulates the exact input preparation logic previously
    inline in :meth:`OpenVINOCausalLM.forward` for PA-transformed models.
    """

    def __init__(self, model: "OpenVINOCausalLM") -> None:
        self.model = model

    def build_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[ov.Tensor, ov.Tensor]],
        ssm_caches: list[ov.Tensor] | None = None,
        conv_caches: list[ov.Tensor] | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
    ) -> list:
        """Build list-based inputs for a PA-transformed model inference request."""
        model = self.model
        state_tensors = model._get_flat_kv_caches_template(kv_caches)
        if model.model_type == HYBRID_MAMBA:
            if ssm_caches:
                state_tensors.extend(ssm_caches)
            if conv_caches:
                state_tensors.extend(conv_caches)

        attn_metadata = get_forward_context().attn_metadata
        block_table_inputs = []
        for block_indices, block_indices_begins in model._iter_block_table_inputs(attn_metadata):
            block_table_inputs.extend((block_indices, block_indices_begins))

        if model.use_text_embeddings_model:
            # Gemma 3 style: language model takes inputs_embeds, not input_ids.
            # Run text embeddings model first: input_ids (1D) → inputs_embeds.
            input_ids_np = model._as_numpy_no_copy(input_ids).reshape(1, -1)
            model.text_emb_request.infer([input_ids_np])
            inputs_embeds = model.text_emb_request.get_output_tensor(0)
            # inputs_embeds shape: [1, seq_len, hidden] → flatten to [seq_len, hidden]
            inputs_embeds_2d = inputs_embeds.data.reshape(-1, inputs_embeds.shape[-1])

            if model.use_vision_embeddings_model and pixel_values is not None:
                pixel_values_np = model._as_numpy_no_copy(pixel_values)
                image_pos_np = model._as_numpy_no_copy(image_position_ids)
                model.vision_emb_request.infer([pixel_values_np, image_pos_np])
                vision_embeds = model.vision_emb_request.get_output_tensor(0)
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
        return inputs


class StatefulInputBuilder(OpenVINOInputBuilder):
    """Builds dict-based inputs for stateful OpenVINO models.

    Stateful models use OpenVINO's internal state management (ReadValue/Assign
    ops) rather than explicit KV cache tensors passed as inputs.
    """

    def __init__(self, model: "OpenVINOCausalLM") -> None:
        self.model = model
        compiled_model = model.ov_request.get_compiled_model()
        self.input_shapes: dict[str, list] = {}
        first_fixed_dims: list[int] = []
        for inp in compiled_model.inputs:
            name = inp.get_any_name()
            shape = []
            for dim in inp.get_partial_shape():
                if dim.is_dynamic:
                    shape.append(None)
                else:
                    shape.append(dim.get_length())
            self.input_shapes[name] = shape
            if len(shape) > 0 and shape[0] is not None:
                first_fixed_dims.append(shape[0])
        if first_fixed_dims:
            from statistics import mode
            try:
                self.batch_size = mode(first_fixed_dims)
            except StatisticsError:
                self.batch_size = max(first_fixed_dims)
        else:
            self.batch_size = 1
        logger.info("StatefulInputBuilder shape registry: %s, batch_size: %d",
                    self.input_shapes, self.batch_size)

    def build_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[ov.Tensor, ov.Tensor]],
        ssm_caches: list[ov.Tensor] | None = None,
        conv_caches: list[ov.Tensor] | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
    ) -> dict:
        """Build dict-based inputs for a stateful model inference request."""
        model = self.model
        inputs_dict: dict[str, np.ndarray] = {}

        if model.use_text_embeddings_model:
            input_ids_np = model._as_numpy_no_copy(input_ids).reshape(1, -1)
            model.text_emb_request.infer([input_ids_np])
            inputs_embeds = model.text_emb_request.get_output_tensor(0)
            # Text embeddings output: [1, seq_len, hidden] -> [seq_len, hidden]
            inputs_embeds_2d = inputs_embeds.data.reshape(
                -1, inputs_embeds.shape[-1])

            if model.use_vision_embeddings_model and pixel_values is not None:
                pixel_values_np = model._as_numpy_no_copy(pixel_values)
                image_pos_np = model._as_numpy_no_copy(image_position_ids)
                model.vision_emb_request.infer(
                    [pixel_values_np, image_pos_np])
                vision_embeds = model.vision_emb_request.get_output_tensor(0)
                vision_embeds_2d = vision_embeds.data.reshape(
                    -1, vision_embeds.shape[-1])

                for i in range(image_pos_np.shape[1]):
                    start, end = image_pos_np[0, i]
                    num_patches = end - start
                    patch_offset = i * num_patches
                    inputs_embeds_2d[start:end] = vision_embeds_2d[
                        patch_offset:patch_offset + num_patches]

            seq_len = inputs_embeds_2d.shape[0]
            hidden = inputs_embeds_2d.shape[1]

            for name, shape in self.input_shapes.items():
                if name == "inputs_embeds":
                    inputs_dict[name] = np.tile(
                        inputs_embeds_2d[np.newaxis, :, :],
                        (self.batch_size, 1, 1))
                elif name == "position_ids":
                    if len(shape) == 3:
                        pos = np.arange(seq_len, dtype=np.int64)
                        inputs_dict[name] = np.tile(
                            pos, (self.batch_size, 1)).reshape(
                                self.batch_size, 1, seq_len)
                    elif len(shape) == 2:
                        pos = model._as_numpy_no_copy(positions)
                        inputs_dict[name] = pos.reshape(self.batch_size, -1)
                    else:
                        inputs_dict[name] = model._as_numpy_no_copy(
                            positions).reshape(-1)
                elif name == "attention_mask":
                    inputs_dict[name] = np.ones(
                        (self.batch_size, seq_len), dtype=np.int64)
                elif name == "beam_idx":
                    inputs_dict[name] = np.arange(
                        self.batch_size, dtype=np.int32)
                else:
                    logger.warning(
                        "StatefulInputBuilder: unhandled input %s "
                        "(shape %s), skipping", name, shape)
        else:
            input_ids_np = model._as_numpy_no_copy(input_ids)
            if input_ids_np.ndim == 1:
                seq_len = input_ids_np.shape[0]
            else:
                seq_len = input_ids_np.shape[1]

            for name, shape in self.input_shapes.items():
                if name == "input_ids":
                    if len(shape) == 2:
                        inputs_dict[name] = input_ids_np.reshape(
                            self.batch_size, -1)
                    else:
                        inputs_dict[name] = input_ids_np.reshape(-1)
                elif name == "inputs_embeds":
                    hidden = shape[-1] if shape[-1] is not None else 2048
                    inputs_dict[name] = np.zeros(
                        (self.batch_size, seq_len, hidden), dtype=np.float32)
                elif name == "position_ids":
                    if len(shape) == 3:
                        pos = np.arange(seq_len, dtype=np.int64)
                        inputs_dict[name] = np.tile(
                            pos, (self.batch_size, 1)).reshape(
                                self.batch_size, 1, seq_len)
                    elif len(shape) == 2:
                        pos = model._as_numpy_no_copy(positions)
                        inputs_dict[name] = pos.reshape(self.batch_size, -1)
                    else:
                        inputs_dict[name] = model._as_numpy_no_copy(
                            positions).reshape(-1)
                elif name == "attention_mask":
                    inputs_dict[name] = np.ones(
                        (self.batch_size, seq_len), dtype=np.int64)
                elif name == "beam_idx":
                    inputs_dict[name] = np.arange(
                        self.batch_size, dtype=np.int32)
                else:
                    logger.warning(
                        "StatefulInputBuilder: unhandled input %s "
                        "(shape %s), skipping", name, shape)

        return inputs_dict


def get_model(
    vllm_config: VllmConfig,
    ov_core: ov.Core,
) -> torch.nn.Module:
    with set_current_vllm_config(vllm_config):
        return OpenVINOCausalLM(ov_core, vllm_config.model_config)

 
