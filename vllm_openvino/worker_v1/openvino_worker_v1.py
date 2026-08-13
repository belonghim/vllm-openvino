# SPDX-License-Identifier: Apache-2.0
import math
from pathlib import Path

import openvino as ov
import openvino.properties as ov_props
import torch
import torch.distributed
import torch.nn as nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils.torch_utils import set_random_seed
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

from vllm.v1.worker.utils import bind_kv_cache
from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheConfig, FullAttentionSpec, MambaSpec
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase, CompilationTimes
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData

import vllm_openvino.envs as envs
from vllm_openvino.worker_v1.openvino_model_runner_v1 import OpenVINOModelRunnerV1
from vllm_openvino.kv_cache import OpenVINOCacheEngine
from vllm_openvino.utils import determine_num_available_blocks, get_max_allocatable_memory_gpu, format_memory_size
from vllm_openvino.model_executor.model_loader.openvino import (
    ATTENTION_ONLY, HYBRID_MAMBA, STATEFUL,
)

logger = init_logger(__name__)

str_to_torch_type: dict[str, torch.dtype] = {
    "u8": torch.uint8,
    "i8": torch.int8,
    "fp16": torch.float16,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "fp32": torch.float32
}

USED_MEMORY_THRESHOLD = 1.1  # 10% overhead for unaccounted memory


def _resolve_cache_dtype(dtype: str | None) -> str:
    # vLLM uses "dynamic" / None to mean "backend chooses"; OV defaults to fp16.
    if dtype is None or dtype == "dynamic":
        return "fp16"
    return dtype


class OpenVINOWorkerV1(WorkerBase):
    """A worker class that executes the model on OpenVINO backend.

    Each worker is associated with a single OpenVINO device. The worker is
    responsible for maintaining the KV cache and executing the model on the
    OpenVINO backend.
    """

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
    ):
        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)
        self.ov_core = ov.Core()
        self.ov_core.set_property({ov_props.enable_mmap: True})
        self.parallel_config.rank = rank

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from transformers.dynamic_module_utils import init_hf_modules

            init_hf_modules()
        self.model_runner = OpenVINOModelRunnerV1(
            vllm_config=self.vllm_config,
            device=self.device,
            ov_core=self.ov_core,
        )

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: OpenVINOCacheEngine
        self.kv_cache: list[tuple[ov.Tensor, ov.Tensor]]
        self.num_swap_blocks = 0
        self._pending_output: "ModelRunnerOutput | None" = None

        # Cache shape metadata (needed before determine_available_memory()).
        self.key_cache_config = []
        self.value_cache_config = []
        self.ssm_cache_config = []
        self.conv_cache_config = []
        self.ssm_cache_dtypes = []
        self.conv_cache_dtypes = []
        self._preloaded_model_type = ATTENTION_ONLY
        self._preloaded_ssm_state_shapes = None
        self._preloaded_ov_model = None

        # Preload SSM/conv cache shapes from model IR so memory sizing includes
        # hybrid-model state tensors even before load_model() is called.
        self._preload_state_cache_shapes()

    def init_device(self) -> None:
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def _preload_state_cache_shapes(self) -> None:
        """Populate SSM/conv cache shapes from IR before memory sizing.

        determine_available_memory() may be called before load_model().
        For hybrid models, we still need SSM/conv shapes for correct
        per-block memory calculation.
        """
        model_dir = Path(self.model_config.model)
        if not model_dir.is_dir():
            return

        if (model_dir / "openvino_language_model.xml").exists():
            ir_path = model_dir / "openvino_language_model.xml"
        elif (model_dir / "openvino_model.xml").exists():
            ir_path = model_dir / "openvino_model.xml"
        else:
            return

        try:
            ov_model = self.ov_core.read_model(str(ir_path))
            ssm_cache_config = []
            conv_cache_config = []
            ssm_cache_dtypes = []
            conv_cache_dtypes = []
            key_cache_config = []
            value_cache_config = []
            cache_dtype = None

            has_unknown_readvalue = False
            for op in ov_model.get_ops():
                if op.get_type_name() != "ReadValue":
                    continue
                var_id = op.get_variable_id()
                if not var_id:
                    has_unknown_readvalue = True
                    continue
                if "ssm" in var_id:
                    ssm_cache_config.append(op.output(0).get_partial_shape())
                    ssm_cache_dtypes.append(op.get_element_type().to_string())
                elif "conv" in var_id:
                    conv_cache_config.append(op.output(0).get_partial_shape())
                    conv_cache_dtypes.append(op.get_element_type().to_string())
                elif "past_key_values" in var_id:
                    if cache_dtype is None:
                        cache_dtype = op.get_element_type().to_string()
                    if ".key" in var_id:
                        key_cache_config.append(op.output(0).get_partial_shape())
                    elif ".value" in var_id:
                        value_cache_config.append(op.output(0).get_partial_shape())
                else:
                    has_unknown_readvalue = True

            self.ssm_cache_config = ssm_cache_config
            self.conv_cache_config = conv_cache_config
            self.ssm_cache_dtypes = ssm_cache_dtypes
            self.conv_cache_dtypes = conv_cache_dtypes
            self.key_cache_config = key_cache_config
            self.value_cache_config = value_cache_config
            if cache_dtype is not None:
                self.cache_dtype = cache_dtype

            if ssm_cache_config or conv_cache_config:
                self._preloaded_model_type = HYBRID_MAMBA
            elif key_cache_config or value_cache_config or has_unknown_readvalue:
                self._preloaded_model_type = STATEFUL
            else:
                self._preloaded_model_type = ATTENTION_ONLY
            self._preloaded_ssm_state_shapes = {
                "ssm": list(zip(ssm_cache_config, ssm_cache_dtypes)),
                "conv": list(zip(conv_cache_config, conv_cache_dtypes)),
            }
            self._preloaded_ov_model = ov_model

            if self.ssm_cache_config or self.conv_cache_config:
                logger.info(
                    "Preloaded hybrid cache shapes from %s: ssm=%d conv=%d",
                    ir_path,
                    len(self.ssm_cache_config),
                    len(self.conv_cache_config),
                )
            if self.key_cache_config or self.value_cache_config:
                logger.info(
                    "Preloaded KV cache shapes from %s: key=%d value=%d",
                    ir_path,
                    len(self.key_cache_config),
                    len(self.value_cache_config),
                )
        except (RuntimeError, FileNotFoundError) as e:
            logger.warning(
                "Failed to preload SSM/conv cache shapes from %s: %r",
                ir_path,
                e,
            )

    def load_model(self):
        self.model_runner.load_model(
            preloaded_model_type=self._preloaded_model_type,
            preloaded_ssm_state_shapes=self._preloaded_ssm_state_shapes,
            preloaded_ov_model=self._preloaded_ov_model,
        )
        # Release reference; loader consumes it and applies in-place PA transformation.
        self._preloaded_ov_model = None

        model = self.model_runner.get_model()
        if hasattr(model, 'warmup'):
            model.warmup()

        compiled_model = model.ov_request.get_compiled_model()

        ssm_shapes = getattr(model, "ssm_state_shapes", {})
        self.ssm_cache_config = [shape for shape, dtype in ssm_shapes.get("ssm", [])]
        self.conv_cache_config = [shape for shape, dtype in ssm_shapes.get("conv", [])]
        self.ssm_cache_dtypes = [dtype for shape, dtype in ssm_shapes.get("ssm", [])]
        self.conv_cache_dtypes = [dtype for shape, dtype in ssm_shapes.get("conv", [])]

        num_cache_groups = 1
        self.model_runner.configure_cache_groups(num_cache_groups)

        has_external_kv = getattr(model, '_has_kv_cache_inputs', False)
        if has_external_kv:
            new_key_cache_config = []
            new_value_cache_config = []
            for input_port in compiled_model.inputs:
                input_name = input_port.get_any_name()
                if input_name.startswith("key_cache."):
                    self.cache_dtype = input_port.get_element_type().to_string()
                    new_key_cache_config.append(input_port.get_partial_shape())
                elif input_name.startswith("value_cache."):
                    new_value_cache_config.append(input_port.get_partial_shape())
            self.key_cache_config = new_key_cache_config
            self.value_cache_config = new_value_cache_config
            logger.info(
                "[OV-WORKER] PA model, key_cache=%d, value_cache=%d, "
                "ssm_cache=%d, conv_cache=%d",
                len(self.key_cache_config), len(self.value_cache_config),
                len(self.ssm_cache_config), len(self.conv_cache_config),
            )
        else:
            logger.info(
                "[OV-WORKER] Stateful model, using preloaded KV shapes: "
                "key=%d, value=%d",
                len(self.key_cache_config), len(self.value_cache_config),
            )

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache. Swappable CPU memory is only
        supported on GPU.

        For CPU, we use the num_gpu_blocks to
        determine how many non-swappable CPU blocks to allocate.
        """

        num_device_blocks = num_gpu_blocks
        num_swap_blocks = num_cpu_blocks

        if current_platform.is_openvino_cpu():
            assert (num_swap_blocks == 0
                    ), f"{type(self)} does not support swappable cache for CPU"

        self._validate_num_blocks(num_device_blocks)
        self.cache_config.num_gpu_blocks = num_device_blocks
        self.cache_config.num_cpu_blocks = num_swap_blocks

        # Initialize the cache.
        self._init_cache_engine()

    def _validate_num_blocks(self, num_blocks: int) -> None:
        """Raise errors if the num_blocks is invalid."""
        if num_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `VLLM_OPENVINO_KVCACHE_SPACE` when "
                "initializing the engine.")

        if self._is_model_stateful():
            return

        max_seq_len = self.cache_config.block_size * num_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                f"is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`VLLM_OPENVINO_KVCACHE_SPACE` or decreasing `max_model_len` "
                "when initializing the engine.")

    def _init_cache_engine(self) -> None:
        ov_device = envs.VLLM_OPENVINO_DEVICE
        detected_dtype = _resolve_cache_dtype(
            getattr(self, 'cache_dtype', None) or self.cache_config.cache_dtype)
        self.cache_config.cache_dtype = detected_dtype

        num_ssm_blocks = None
        is_stateful = self._is_model_stateful()
        if is_stateful and self.ssm_cache_config:
            num_ssm_blocks = self.scheduler_config.max_num_seqs + 1
            logger.info(
                "[OV-WORKER] Stateful model: SSM physical slots=%d "
                "(scheduler blocks=%d)",
                num_ssm_blocks, self.cache_config.num_gpu_blocks,
            )

        kv_key_config = [] if is_stateful else self.key_cache_config
        kv_value_config = [] if is_stateful else self.value_cache_config

        self.cache_engine = OpenVINOCacheEngine(
            self.cache_config,
            kv_key_config,
            kv_value_config,
            self.model_config,
            self.parallel_config,
            self.device_config,
            self.ov_core,
            ov_device,
            self.ssm_cache_config,
            self.conv_cache_config,
            self.ssm_cache_dtypes,
            self.conv_cache_dtypes,
            num_ssm_blocks,
        )
        self.kv_cache = self.cache_engine.kv_cache
        self.model_runner.kv_caches = self.kv_cache
        self.model_runner.ssm_caches = self.cache_engine.ssm_cache
        self.model_runner.conv_caches = self.cache_engine.conv_cache
        bind_kv_cache({}, self.compilation_config.static_forward_context, [])
        self.model_runner.block_size = self.cache_engine.block_size

        assert self.kv_cache is not None

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        if scheduler_output.total_num_scheduled_tokens == 0:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=None,
            )
        new_block_ids_to_zero = getattr(scheduler_output, 'new_block_ids_to_zero', None)
        if new_block_ids_to_zero and self.kv_cache:
            for key_cache, value_cache in self.kv_cache:
                k = key_cache.data
                v = value_cache.data
                valid_ids = [bid for bid in new_block_ids_to_zero if bid < k.shape[0]]
                if valid_ids:
                    k[valid_ids] = 0
                    v[valid_ids] = 0
        self._pending_output = self.model_runner.execute_model(scheduler_output)
        return None

    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        # Structured outputs not supported; sampling is done in execute_model.
        return self._pending_output

    def init_distributed_environment(self) -> None:
        """Initialize the distributed environment."""

        parallel_config = self.parallel_config
        rank = self.rank
        distributed_init_method = self.distributed_init_method
        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            backend="gloo",
        )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cpu())

        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
        )

    def get_cache_block_size_bytes(self) -> int:
        """Return the size in bytes of a single KV cache block."""
        return OpenVINOCacheEngine.get_cache_block_size(
            self.cache_config.cache_dtype,
            self.key_cache_config,
            self.value_cache_config,
            self.ssm_cache_config,
            self.conv_cache_config,
            self.cache_config.block_size,
        )

    def profile_run(self) -> int:
        ov_device = envs.VLLM_OPENVINO_DEVICE

        assert not current_platform.is_openvino_cpu(), \
            "CPU device isn't supposed to use profile run."

        import openvino.properties.device as device
        import openvino.properties.intel_gpu as intel_gpu

        ov_core = self.ov_core
        cache_config = self.cache_config
        model_config = self.model_config
        parallel_config = self.parallel_config
        device_config = self.device_config
        mm_registry = MULTIMODAL_REGISTRY

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        def model_profile_run():
            top_k = model_config.get_vocab_size() - 1
            sampling_params = SamplingParams(top_p=0.99, top_k=top_k)

            max_num_batched_tokens = \
                self.scheduler_config.max_num_batched_tokens
            max_num_seqs = self.scheduler_config.max_num_seqs
            tmp_cache_config = CacheConfig(cache_config.block_size,
                                           cache_config.gpu_memory_utilization,
                                           cache_config.swap_space_bytes,
                                           "auto")
            tmp_cache_config.num_gpu_blocks = 1
            tmp_cache_config.num_cpu_blocks = 0
            tmp_cache_config.cache_dtype = cache_config.cache_dtype

            profiling_cache_engine = OpenVINOCacheEngine(
                tmp_cache_config,
                self.key_cache_config,
                self.value_cache_config,
                model_config,
                parallel_config,
                device_config,
                ov_core,
                ov_device,
                self.ssm_cache_config,
                self.conv_cache_config,
                self.ssm_cache_dtypes,
                self.conv_cache_dtypes,
            )
            prev_kv_caches = self.model_runner.kv_caches
            prev_ssm_caches = getattr(self.model_runner, 'ssm_caches', None)
            prev_conv_caches = getattr(self.model_runner, 'conv_caches', None)
            self.model_runner.kv_caches = profiling_cache_engine.kv_cache
            self.model_runner.ssm_caches = profiling_cache_engine.ssm_cache
            self.model_runner.conv_caches = profiling_cache_engine.conv_cache

            total_num_scheduled_tokens = 0
            num_scheduled_tokens = {}
            reqs = []
            block_size = cache_config.block_size
            num_blocks = 0

            for group_id in range(max_num_seqs):
                seq_len = (max_num_batched_tokens // max_num_seqs +
                           (group_id < max_num_batched_tokens % max_num_seqs))
                seq_num_blocks = (seq_len + block_size - 1) // block_size

                dummy_data = mm_registry.get_decoder_dummy_data(model_config, seq_len)

                block_table = list(range(num_blocks, num_blocks + seq_num_blocks))
                num_blocks += seq_num_blocks
                reqs.append(NewRequestData(
                    req_id=str(group_id),
                    prompt_token_ids=list(dummy_data.seq_data.prompt_token_ids),
                    mm_features=[],
                    sampling_params=sampling_params,
                    pooling_params=None,
                    block_ids=(block_table,),
                    num_computed_tokens=0,
                    lora_request=None,
                ))
                num_scheduled_tokens[str(group_id)] = seq_len
                total_num_scheduled_tokens += seq_len

            from vllm.v1.core.sched.output import CachedRequestData
            scheduler_output = SchedulerOutput(
                scheduled_new_reqs=reqs,
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens=num_scheduled_tokens,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[0],
                finished_req_ids=set(),
                free_encoder_mm_hashes=[],
            )
            self.model_runner.block_size = tmp_cache_config.block_size

            bind_kv_cache({}, self.compilation_config.static_forward_context, [])
            try:
                self.model_runner.execute_model(scheduler_output)
            finally:
                bind_kv_cache({}, self.compilation_config.static_forward_context, [])
                self.model_runner.kv_caches = prev_kv_caches
                self.model_runner.ssm_caches = prev_ssm_caches
                self.model_runner.conv_caches = prev_conv_caches
                del profiling_cache_engine

        logger.info(
            "Start profiling run with dummy inputs to evaluate "
            "memory usage for %s. It might take a while.", ov_device)
        model_profile_run()

        gpu_device_type = ov_core.get_property(ov_device, device.type)
        memory_statistics = \
            ov_core.get_property(ov_device, intel_gpu.memory_statistics)
        memory_utilization = cache_config.gpu_memory_utilization

        if gpu_device_type == device.Type.INTEGRATED and \
            memory_utilization >= 0.9:
            logger.warning(
                "iGPU is used with high gpu_memory_utilization=%f "
                "value. This may cause low performance due to "
                "occupying the majority of available system "
                "memory. Please consider decreasing "
                "gpu_memory_utilization or explicitly setting "
                "`VLLM_OPENVINO_KVCACHE_SPACE` (GB) environment "
                "variable.", memory_utilization)

        # sum up all used device memory
        device_memory_types = ["cl_mem", "usm_device"]
        used_device_mem = \
            sum(memory_statistics.get(key, 0) for key in device_memory_types)

        if gpu_device_type == device.Type.INTEGRATED:
            used_device_mem += memory_statistics.get("usm_host", 0)

        # there could be unaccounted extra memory reserved by kernels, kept
        # in memory pools, etc
        # therefore, add a threshold to account for this
        used_memory_threshold = USED_MEMORY_THRESHOLD
        used_device_mem *= used_memory_threshold

        total_device_memory = \
            ov_core.get_property(ov_device, intel_gpu.device_total_mem_size)

        total_device_memory_str = format_memory_size(total_device_memory)
        used_device_memory_str = format_memory_size(used_device_mem)

        logger.info(
            "Total %s memory: %s. "
            "Amount of memory required to run the model with "
            "max_num_batched_tokens=%d: %s.", ov_device,
            total_device_memory_str,
            self.scheduler_config.max_num_batched_tokens,
            used_device_memory_str)

        if used_device_mem >= total_device_memory * memory_utilization:
            raise RuntimeError(
                f"The required memory size {used_device_memory_str} for model "
                f"is higher than the available device "
                f"memory {total_device_memory_str} * {memory_utilization} utilization. "
                f"Please consider to decrease `max_num_batched_tokens` or increase "
                f"`gpu_memory_utilization`")

        # Reset input batch
        self.model_runner.configure_cache_groups(self.model_runner.num_cache_groups)

        available_memory = total_device_memory * memory_utilization - used_device_mem
        return min(available_memory, get_max_allocatable_memory_gpu(ov_core, ov_device, self.key_cache_config, self.value_cache_config))

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        key_cache_config = self.key_cache_config
        value_cache_config = self.value_cache_config
        block_size = self.cache_config.block_size
        cache_type = _resolve_cache_dtype(self.cache_config.cache_dtype)
        assert cache_type in str_to_torch_type, f"Unexpected cache type {cache_type}"
        kv_cache_spec = {}

        logger.info(
            "[OV-WORKER] get_kv_cache_spec: key=%d, value=%d, "
            "ssm=%d, conv=%d, block_size=%d, cache_type=%s",
            len(key_cache_config), len(value_cache_config),
            len(self.ssm_cache_config), len(self.conv_cache_config),
            block_size, cache_type,
        )

        for idx, (key_cache_shape, value_cache_shape) in enumerate(zip(key_cache_config, value_cache_config)):
            kv_cache_spec[str(idx)] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=max(key_cache_shape[1].get_length(), value_cache_shape[1].get_length()),
                head_size=max(key_cache_shape[3].get_length(), value_cache_shape[3].get_length()),
                dtype=str_to_torch_type[cache_type])

        # Hybrid models: include MambaSpec entries so vLLM accounts for
        # SSM/conv state memory when allocating cache blocks.
        if self.ssm_cache_config or self.conv_cache_config:
            if (self.ssm_cache_config and self.conv_cache_config
                    and len(self.ssm_cache_config) != len(self.conv_cache_config)):
                logger.warning(
                    "Mismatched SSM/conv cache shapes: ssm=%d conv=%d; using max count.",
                    len(self.ssm_cache_config),
                    len(self.conv_cache_config),
                )

            if self._is_model_stateful():
                # block_size=max_model_len makes ceil(max_model_len/block_size)=1, satisfying full_sequence_must_fit with 1 slot.
                mamba_block_size = self.model_config.max_model_len
            else:
                mamba_block_size = getattr(self.cache_config, "mamba_block_size", None) or block_size
            mamba_cache_mode = getattr(self.cache_config, "mamba_cache_mode", "none")
            if mamba_cache_mode != "none":
                logger.warning(
                    "OpenVINO hybrid models support only mamba_cache_mode='none'; overriding '%s'.",
                    mamba_cache_mode,
                )
                mamba_cache_mode = "none"

            def _to_per_block_shape(pshape: ov.PartialShape) -> tuple[int, ...]:
                # dim0 is num_blocks and excluded from per-page state shape.
                dims = []
                for dim in pshape[1:]:
                    if dim.is_dynamic:
                        dims.append(block_size)
                    else:
                        dims.append(dim.get_length())
                return tuple(dims)

            def _dtype_to_torch(dtype_str: str) -> torch.dtype:
                return str_to_torch_type.get(dtype_str, torch.float32)

            def _get_state_shape_dtype(
                cache_config: list,
                cache_dtypes: list,
                idx: int,
            ) -> tuple[tuple[int, ...], torch.dtype]:
                if idx < len(cache_config):
                    dtype = _dtype_to_torch(cache_dtypes[idx]) if idx < len(cache_dtypes) else torch.float32
                    return _to_per_block_shape(cache_config[idx]), dtype
                return (1,), torch.float32

            num_mamba_layers = max(len(self.ssm_cache_config), len(self.conv_cache_config))
            for i in range(num_mamba_layers):
                conv_shape, conv_dtype = _get_state_shape_dtype(self.conv_cache_config, self.conv_cache_dtypes, i)
                ssm_shape, ssm_dtype = _get_state_shape_dtype(self.ssm_cache_config, self.ssm_cache_dtypes, i)

                # MambaSpec.shapes expects tuple[tuple[int, ...], ...], e.g.
                # (conv_state_shape, ssm_state_shape).
                kv_cache_spec[f"mamba.{i}"] = MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(conv_shape, ssm_shape),
                    dtypes=(conv_dtype, ssm_dtype),
                    mamba_type=MambaAttentionBackendEnum.MAMBA2,
                    mamba_cache_mode=mamba_cache_mode,
                    num_speculative_blocks=0,
                )

        return kv_cache_spec

    def determine_available_memory(self) -> int:
        """Determines how much memory is needed for KV-cache
        """
        cache_dtype = _resolve_cache_dtype(
            getattr(self, 'cache_dtype', None) or self.cache_config.cache_dtype)
        self.cache_config.cache_dtype = cache_dtype
        cache_block_size = self.get_cache_block_size_bytes()
        kv_space = getattr(self.cache_config, 'openvino_kvcache_space_bytes', 0)
        logger.info(
            "[OV-WORKER] determine_available_memory: cache_dtype=%s, "
            "cache_block_size=%d bytes, kv_space=%d bytes",
            cache_dtype, cache_block_size, kv_space,
        )
        num_device_blocks, num_swap_blocks = determine_num_available_blocks(current_platform,
                                                                            self.cache_config,
                                                                            cache_block_size,
                                                                            self.profile_run)
        if self._is_model_stateful():
            blocks_per_seq = math.ceil(
                self.model_config.max_model_len / self.cache_config.block_size)
            max_needed = blocks_per_seq * self.scheduler_config.max_num_seqs + 1
            if num_device_blocks > max_needed:
                logger.info(
                    "[OV-WORKER] Capping stateful num_blocks %d -> %d",
                    num_device_blocks, max_needed)
                num_device_blocks = max_needed
        logger.info(
            "[OV-WORKER] determine_available_memory: num_device_blocks=%d, "
            "num_swap_blocks=%d",
            num_device_blocks, num_swap_blocks,
        )
        self.num_swap_blocks = num_swap_blocks
        return num_device_blocks * cache_block_size

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate OpenVINO KV cache with the specified kv_cache_config."""
        self.initialize_cache(kv_cache_config.num_blocks, self.num_swap_blocks)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # OpenVINO compiles in load_model(); callers ignore the return value.
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def update_max_model_len(self, max_model_len: int) -> None:
        self.model_config.max_model_len = max_model_len

    def _is_model_stateful(self) -> bool:
        model = getattr(self.model_runner, 'model', None)
        if model is not None:
            return not getattr(model, '_has_kv_cache_inputs', True)
        # Fallback before load_model() completes (e.g. determine_available_memory called early)
        preloaded = getattr(self, '_preloaded_model_type', None)
        return preloaded in (STATEFUL, HYBRID_MAMBA)

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ('generate',)

    def list_loras(self) -> set[int]:
        raise NotImplementedError("LoRA is not supported.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def shutdown(self) -> None:
        logger.info("[OV-WORKER] Shutting down OpenVINO worker")
        model = getattr(self.model_runner, 'model', None)
        if model is not None and hasattr(model, 'shutdown'):
            model.shutdown()

    def determine_num_available_blocks(self) -> tuple[int, int]:
        raise NotImplementedError(
            "Use determine_available_memory() instead. "
            "This method is not used in V1 engine."
        )
