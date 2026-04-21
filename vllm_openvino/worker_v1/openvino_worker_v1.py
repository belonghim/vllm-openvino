# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import List, Tuple, Set

import openvino as ov
import torch
import torch.distributed
import torch.nn as nn
from vllm.config import (CacheConfig, VllmConfig)
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
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData

import vllm_openvino.envs as envs
from vllm_openvino.worker_v1.openvino_model_runner_v1 import OpenVINOModelRunnerV1
from vllm_openvino.kv_cache import OpenVINOCacheEngine
from vllm_openvino.utils import determine_num_available_blocks, get_max_allocatable_memory_gpu
from vllm_openvino.model_executor.model_loader.openvino import ATTENTION_ONLY, HYBRID_MAMBA

logger = init_logger(__name__)

str_to_torch_type = {
    "u8": torch.uint8,
    "i8": torch.int8,
    "fp16": torch.float16,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "fp32": torch.float32
}


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
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

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
        self.kv_cache: List[Tuple[ov.Tensor, ov.Tensor]]
        self.num_swap_blocks = 0

        # Cache shape metadata (needed before determine_available_memory()).
        self.key_cache_config = []
        self.value_cache_config = []
        self.ssm_cache_config = []
        self.conv_cache_config = []
        self.ssm_cache_dtypes = []
        self.conv_cache_dtypes = []

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

            for op in ov_model.get_ops():
                if op.get_type_name() != "ReadValue":
                    continue
                var_id = op.get_variable_id()
                if not var_id:
                    continue
                if "ssm" in var_id:
                    ssm_cache_config.append(op.output(0).get_partial_shape())
                    ssm_cache_dtypes.append(op.get_element_type().to_string())
                elif "conv" in var_id:
                    conv_cache_config.append(op.output(0).get_partial_shape())
                    conv_cache_dtypes.append(op.get_element_type().to_string())

            self.ssm_cache_config = ssm_cache_config
            self.conv_cache_config = conv_cache_config
            self.ssm_cache_dtypes = ssm_cache_dtypes
            self.conv_cache_dtypes = conv_cache_dtypes
            if self.ssm_cache_config or self.conv_cache_config:
                logger.info(
                    "Preloaded hybrid cache shapes from %s: ssm=%d conv=%d",
                    ir_path,
                    len(self.ssm_cache_config),
                    len(self.conv_cache_config),
                )
        except Exception as e:
            logger.warning(
                "Failed to preload SSM/conv cache shapes from %s: %r",
                ir_path,
                e,
            )

    def load_model(self):
        self.model_runner.load_model()

        # we need to take information about KV cache config from compiled model
        compiled_model = self.model_runner.get_model().ov_request.get_compiled_model()

        self.key_cache_config = []
        self.value_cache_config = []
        self.ssm_cache_config = []
        self.conv_cache_config = []
        self.ssm_cache_dtypes = []
        self.conv_cache_dtypes = []

        ov_model_obj = self.model_runner.get_model()
        ssm_shapes = getattr(ov_model_obj, "ssm_state_shapes", {})
        self.ssm_cache_config = [shape for shape, dtype in ssm_shapes.get("ssm", [])]
        self.conv_cache_config = [shape for shape, dtype in ssm_shapes.get("conv", [])]
        self.ssm_cache_dtypes = [dtype for shape, dtype in ssm_shapes.get("ssm", [])]
        self.conv_cache_dtypes = [dtype for shape, dtype in ssm_shapes.get("conv", [])]

        num_cache_groups = 1
        self.model_runner.configure_cache_groups(num_cache_groups)

        for input_port in compiled_model.inputs:
            input_name = input_port.get_any_name()

            if input_name.startswith("key_cache."):
                self.cache_dtype = input_port.get_element_type().to_string()
                self.key_cache_config.append(input_port.get_partial_shape())
            if input_name.startswith("value_cache."):
                self.value_cache_config.append(input_port.get_partial_shape())

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

        max_seq_len = self.cache_config.block_size * num_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`VLLM_OPENVINO_KVCACHE_SPACE` or decreasing `max_model_len` "
                "when initializing the engine.")

    def _init_cache_engine(self) -> None:
        ov_device = envs.VLLM_OPENVINO_DEVICE
        # Override precision in self.cache_config to one used during compile_model
        # Use cache_dtype from config, falling back to dynamically detected value
        detected_dtype = getattr(self, 'cache_dtype', None)
        if detected_dtype is None:
            detected_dtype = self.cache_config.cache_dtype
        if detected_dtype == "dynamic":
            detected_dtype = "fp16"
        self.cache_config.cache_dtype = detected_dtype

        self.cache_engine = OpenVINOCacheEngine(
            self.cache_config,
            self.key_cache_config,
            self.value_cache_config,
            self.model_config,
            self.parallel_config,
            self.device_config,
            self.ov_core,
            ov_device,
            self.ssm_cache_config,
            self.conv_cache_config,
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
                ov_device)
            prev_kv_caches = self.model_runner.kv_caches
            self.model_runner.kv_caches = profiling_cache_engine.kv_cache

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
            # Run the model with the dummy inputs.
            self.model_runner.execute_model(scheduler_output)

            # Explicitly revert bind_kv_cache and delete temporary KV cache
            # manager to free KV cache when real inputs will be passed to OV
            bind_kv_cache({}, self.compilation_config.static_forward_context, [])
            self.model_runner.kv_caches = prev_kv_caches
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
        used_memory_threshold = 1.1
        used_device_mem *= used_memory_threshold

        total_device_memory = \
            ov_core.get_property(ov_device, intel_gpu.device_total_mem_size)

        def format_memory_size(size) -> str:
            units = ["B", "KB", "MB", "GB"]
            unit_index = 0

            while size > 1024 and unit_index < len(units) - 1:
                size /= 1024
                unit_index += 1

            return f"{size:.2f} {units[unit_index]}"

        total_device_memory_str = format_memory_size(total_device_memory)
        used_device_memory_str = format_memory_size(used_device_mem)

        logger.info(
            "Total %s memory: %s. "
            "Amount of memory required to run the model with "
            "max_num_batched_tokens=%d: %s.", ov_device,
            total_device_memory_str,
            self.scheduler_config.max_num_batched_tokens,
            used_device_memory_str)

        if used_device_mem >= total_device_memory:
            raise RuntimeError(
                f"The required memory size {used_device_memory_str} for model "
                f"is higher than the total available device "
                f"memory {total_device_memory_str}. Please consider to "
                f"decrease `max_num_batched_tokens` or increase "
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
        cache_type = self.cache_config.cache_dtype
        # Handle "dynamic" dtype - convert to fp16 as default for OpenVINO
        if cache_type == "dynamic":
            cache_type = "fp16"
        assert cache_type in str_to_torch_type.keys(), "Unexpected cache type {}".format(cache_type)
        kv_cache_spec = {}

        for idx, (key_cache_shape, value_cache_shape) in enumerate(zip(key_cache_config, value_cache_config)):
            kv_cache_spec["{}".format(idx)] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=max(key_cache_shape[1].get_length(), value_cache_shape[1].get_length()),
                head_size=max(key_cache_shape[3].get_length(), value_cache_shape[3].get_length()),
                dtype=str_to_torch_type[cache_type])

        # Hybrid models: include MambaSpec entries so vLLM accounts for
        # SSM/conv state memory when allocating cache blocks.
        if self.ssm_cache_config and self.conv_cache_config:
            if len(self.ssm_cache_config) != len(self.conv_cache_config):
                logger.warning(
                    "Mismatched SSM/conv cache shapes: ssm=%d conv=%d; using min count.",
                    len(self.ssm_cache_config),
                    len(self.conv_cache_config),
                )

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

            num_mamba_layers = min(len(self.ssm_cache_config), len(self.conv_cache_config))
            for i in range(num_mamba_layers):
                conv_shape = _to_per_block_shape(self.conv_cache_config[i])
                ssm_shape = _to_per_block_shape(self.ssm_cache_config[i])

                conv_dtype = (
                    _dtype_to_torch(self.conv_cache_dtypes[i])
                    if i < len(self.conv_cache_dtypes)
                    else torch.float32
                )
                ssm_dtype = (
                    _dtype_to_torch(self.ssm_cache_dtypes[i])
                    if i < len(self.ssm_cache_dtypes)
                    else torch.float32
                )

                # MambaSpec.shapes expects tuple[tuple[int, ...], ...], e.g.
                # (conv_state_shape, ssm_state_shape).
                kv_cache_spec[f"mamba.{i}"] = MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(conv_shape, ssm_shape),
                    dtypes=(conv_dtype, ssm_dtype),
                    mamba_type="mamba2",
                    mamba_cache_mode=mamba_cache_mode,
                    num_speculative_blocks=0,
                )

        return kv_cache_spec

    def determine_available_memory(self) -> int:
        """Determines how much memory is needed for KV-cache
        """
        cache_dtype = getattr(self, 'cache_dtype', None)
        if cache_dtype is None:
            cache_dtype = self.cache_config.cache_dtype
        if cache_dtype == "dynamic":
            cache_dtype = "fp16"
        self.cache_config.cache_dtype = cache_dtype
        # For OpenVINO backend, in case of CPU device, the block number will be
        # calculated based on the openvino_kvcache_space_bytes.
        cache_block_size = self.get_cache_block_size_bytes()
        kv_space = getattr(self.cache_config, 'openvino_kvcache_space_bytes', 0)
        logger.info("OpenVINO KV cache space: %d bytes, block_size: %d bytes",
                    kv_space, cache_block_size)
        num_device_blocks, num_swap_blocks = determine_num_available_blocks(current_platform,
                                                                            self.cache_config,
                                                                            cache_block_size,
                                                                            self.profile_run)
        logger.info("OpenVINO num_device_blocks: %d", num_device_blocks)
        self.num_swap_blocks = num_swap_blocks
        return num_device_blocks * cache_block_size

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        self.initialize_cache(kv_cache_config.num_blocks, self.num_swap_blocks)

    def compile_or_warm_up_model(self) -> None:
        # Compile is performed on model loading stage
        pass

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ('generate',)

    def list_loras(self) -> Set[int]:
        raise NotImplementedError("LoRA is not supported.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        raise NotImplementedError(
            "Use determine_available_memory() instead. "
            "This method is not used in V1 engine."
        )
