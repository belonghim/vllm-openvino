# SPDX-License-Identifier: Apache-2.0
"""An OpenVINO KV cache implementation for V1 KVCache interface."""
from vllm_openvino.attention.backends.openvino import OpenVINOAttentionBackend
from vllm_openvino import envs
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    import openvino as ov
except ImportError:
    ov = None  # type: ignore[assignment]

str_to_ov_type: dict[str, ov.Type] = {
    "u8": ov.Type.u8,
    "i8": ov.Type.i8,
    "f16": ov.Type.f16,
    "bf16": ov.Type.bf16,
    "f32": ov.Type.f32,
    "dynamic": ov.Type.dynamic,
} if ov is not None else {}

class OpenVINOCacheEngine:
    """Manages the KV cache for OpenVINO backend, implementing the V1 KVCache interface.

    This class is responsible for initializing and managing CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        key_cache_config: list[ov.PartialShape],
        value_cache_config: list[ov.PartialShape],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        ov_core: ov.Core,
        ov_device: str,
        ssm_cache_config: list[ov.PartialShape] | None = None,
        conv_cache_config: list[ov.PartialShape] | None = None,
        ssm_cache_dtypes: list[str] | None = None,
        conv_cache_dtypes: list[str] | None = None,
    ) -> None:
        # vLLM's device_config.device_type is always "cpu" for OpenVINO backend,
        # even when VLLM_OPENVINO_DEVICE targets GPU (device selection managed separately).
        assert device_config.device_type == "cpu"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.key_cache_config = key_cache_config
        self.value_cache_config = value_cache_config
        self.ssm_cache_config = ssm_cache_config if ssm_cache_config is not None else []
        self.conv_cache_config = conv_cache_config if conv_cache_config is not None else []
        self.ssm_cache_dtypes = ssm_cache_dtypes if ssm_cache_dtypes is not None else []
        self.conv_cache_dtypes = conv_cache_dtypes if conv_cache_dtypes is not None else []
        self.num_layers = len(self.value_cache_config)

        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for OpenVINO backend with a CPU target device, because we want
        # to reuse KV cache management in the scheduler.
        self.num_device_blocks = cache_config.num_gpu_blocks
        self.num_swap_blocks = cache_config.num_cpu_blocks

        # OpenVINO uses its own attention backend directly (no vLLM standard backend needed).

        cache_dtype = self.cache_config.cache_dtype
        normalized = envs.KV_CACHE_PRECISION_MAP.get(cache_dtype, cache_dtype)
        if normalized not in str_to_ov_type:
            raise ValueError(
                f"Invalid cache_dtype '{cache_dtype}' for OpenVINO backend. "
                f"Valid options are: {list(str_to_ov_type.keys())}"
            )
        self.ov_cache_dtype = str_to_ov_type[normalized]

        # Initialize the cache.
        self.kv_cache: list[tuple[ov.Tensor, ov.Tensor]] = self._allocate_kv_cache(
            self.num_device_blocks, ov_core,
            ov_device)

        # Initialize SSM/conv state caches (for hybrid models like Mamba).
        self.ssm_cache: list[ov.Tensor] = self._allocate_ssm_cache(
            self.num_device_blocks, ov_core, ov_device)
        self.conv_cache: list[ov.Tensor] = self._allocate_conv_cache(
            self.num_device_blocks, ov_core, ov_device)

        # Initialize the swap.
        self.swap_cache: list[tuple[ov.Tensor, ov.Tensor]] = self._allocate_swap_cache(
            self.num_swap_blocks, ov_device)

        # Cache k_cache and v_cache lists to avoid rebuilding on every call.
        # self.kv_cache structure is immutable after init (only tensor data changes),
        # so caching is safe.
        self._k_cache: list[ov.Tensor] = [tensor[0] for tensor in self.kv_cache]
        self._v_cache: list[ov.Tensor] = [tensor[1] for tensor in self.kv_cache]

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        ov_core: ov.Core,
        ov_device: str,
    ) -> list[tuple[ov.Tensor, ov.Tensor]]:
        """Allocates KV cache."""
        kv_cache: list[tuple[ov.Tensor, ov.Tensor]] = []

        for key_cache_pshape, value_cache_pshape in zip(self.key_cache_config, self.value_cache_config):
            key_dims = [
                num_blocks if i == 0 else (dim.get_length() if dim.is_static else self.block_size)
                for i, dim in enumerate(key_cache_pshape)
            ]
            key_cache_shape = ov.PartialShape(key_dims).to_shape()

            value_dims = [
                num_blocks if i == 0 else (dim.get_length() if dim.is_static else self.block_size)
                for i, dim in enumerate(value_cache_pshape)
            ]
            value_cache_shape = ov.PartialShape(value_dims).to_shape()

            if current_platform.is_openvino_cpu():
                key_blocks = ov.Tensor(self.ov_cache_dtype, key_cache_shape)
                value_blocks = ov.Tensor(self.ov_cache_dtype, value_cache_shape)
                kv_cache.append((key_blocks, value_blocks))
            else:
                remote_context = ov_core.get_default_context(ov_device)
                key_blocks = remote_context.create_tensor(self.ov_cache_dtype, key_cache_shape, {})
                value_blocks = remote_context.create_tensor(self.ov_cache_dtype, value_cache_shape, {})
                kv_cache.append((key_blocks, value_blocks))

        return kv_cache

    def _allocate_ssm_cache(
        self,
        num_blocks: int,
        ov_core: ov.Core,
        ov_device: str,
    ) -> list[ov.Tensor]:
        """Allocates SSM state cache for hybrid models."""
        ssm_cache: list[ov.Tensor] = []

        for idx, ssm_pshape in enumerate(self.ssm_cache_config):
            ssm_dims = [
                num_blocks if i == 0 else (dim.get_length() if dim.is_static else 1)
                for i, dim in enumerate(ssm_pshape)
            ]
            ssm_shape = ov.PartialShape(ssm_dims).to_shape()
            dtype_str = self.ssm_cache_dtypes[idx] if idx < len(self.ssm_cache_dtypes) else "f32"
            ssm_ov_type = str_to_ov_type.get(dtype_str, ov.Type.f32)

            if current_platform.is_openvino_cpu():
                ssm_tensor = ov.Tensor(ssm_ov_type, ssm_shape)
                ssm_tensor.data.fill(0)
            else:
                remote_context = ov_core.get_default_context(ov_device)
                ssm_tensor = remote_context.create_tensor(ssm_ov_type, ssm_shape, {})
            ssm_cache.append(ssm_tensor)

        return ssm_cache

    def _allocate_conv_cache(
        self,
        num_blocks: int,
        ov_core: ov.Core,
        ov_device: str,
    ) -> list[ov.Tensor]:
        """Allocates conv state cache for hybrid models."""
        conv_cache: list[ov.Tensor] = []

        for idx, conv_pshape in enumerate(self.conv_cache_config):
            conv_dims = [
                num_blocks if i == 0 else (dim.get_length() if dim.is_static else 1)
                for i, dim in enumerate(conv_pshape)
            ]
            conv_shape = ov.PartialShape(conv_dims).to_shape()
            dtype_str = self.conv_cache_dtypes[idx] if idx < len(self.conv_cache_dtypes) else "f32"
            conv_ov_type = str_to_ov_type.get(dtype_str, ov.Type.f32)

            if current_platform.is_openvino_cpu():
                conv_tensor = ov.Tensor(conv_ov_type, conv_shape)
                conv_tensor.data.fill(0)
            else:
                remote_context = ov_core.get_default_context(ov_device)
                conv_tensor = remote_context.create_tensor(conv_ov_type, conv_shape, {})
            conv_cache.append(conv_tensor)

        return conv_cache

    def _allocate_swap_cache(
        self,
        num_blocks: int,
        ov_device: str,
    ) -> list[tuple[ov.Tensor, ov.Tensor]]:
        """Allocates swap cache."""
        swap_cache: list[tuple[ov.Tensor, ov.Tensor]] = []

        if num_blocks == 0:
            return swap_cache

        assert not current_platform.is_openvino_cpu(), \
            "CPU device isn't supposed to have swap cache"

        for key_cache_pshape, value_cache_pshape in zip(self.key_cache_config, self.value_cache_config):
            key_dims = [
                num_blocks if i == 0 else (dim.get_length() if dim.is_static else self.block_size)
                for i, dim in enumerate(key_cache_pshape)
            ]
            value_dims = [
                num_blocks if i == 0 else (dim.get_length() if dim.is_static else self.block_size)
                for i, dim in enumerate(value_cache_pshape)
            ]
            key_blocks = ov.Tensor(self.ov_cache_dtype, ov.PartialShape(key_dims).to_shape())
            value_blocks = ov.Tensor(self.ov_cache_dtype, ov.PartialShape(value_dims).to_shape())
            swap_cache.append((key_blocks, value_blocks))

        return swap_cache

    def swap_in(self, src_to_dst: list[tuple[int, int]]) -> None:
        for i in range(self.num_layers):
            for swap_tensor, kv_tensor in zip(self.swap_cache[i],
                                              self.kv_cache[i]):
                OpenVINOAttentionBackend.swap_blocks(swap_tensor, kv_tensor,
                                                    src_to_dst)

    def swap_out(self, src_to_dst: list[tuple[int, int]]) -> None:
        for i in range(self.num_layers):
            for swap_tensor, kv_tensor in zip(self.swap_cache[i],
                                              self.kv_cache[i]):
                OpenVINOAttentionBackend.swap_blocks(kv_tensor, swap_tensor,
                                                    src_to_dst)

    def copy(self, src_to_dsts: list[tuple[int, int]]) -> None:
        if len(src_to_dsts) > 0:
            OpenVINOAttentionBackend.copy_blocks(self.kv_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_dtype: str,
        key_cache_config: list[ov.PartialShape],
        value_cache_config: list[ov.PartialShape],
        ssm_cache_config: list[ov.PartialShape] | None = None,
        conv_cache_config: list[ov.PartialShape] | None = None,
        block_size: int = 1,
    ) -> int:
        normalized = envs.KV_CACHE_PRECISION_MAP.get(cache_dtype, cache_dtype)

        def _dim_len(dim, idx: int = -1):
            if dim.is_static:
                return dim.get_length()
            if idx == 2:
                return block_size
            return 1

        total_elements = 0
        for key_cache_shape, value_cache_shape in zip(key_cache_config, value_cache_config):
            total_elements += _dim_len(key_cache_shape[1]) * _dim_len(key_cache_shape[2], 2) * _dim_len(key_cache_shape[3])
            total_elements += _dim_len(value_cache_shape[1]) * _dim_len(value_cache_shape[2], 2) * _dim_len(value_cache_shape[3])

        # Add SSM state size (fp32 = 4 bytes)
        if ssm_cache_config:
            for ssm_shape in ssm_cache_config:
                ssm_elements = 1
                for dim_idx, dim in enumerate(ssm_shape[1:], start=1):
                    ssm_elements *= _dim_len(dim, dim_idx)
                total_elements += ssm_elements * (4 / str_to_ov_type[normalized].size)

        # Add conv state size (fp32 = 4 bytes)
        if conv_cache_config:
            for conv_shape in conv_cache_config:
                conv_elements = 1
                for dim_idx, dim in enumerate(conv_shape[1:], start=1):
                    conv_elements *= _dim_len(dim, dim_idx)
                total_elements += conv_elements * (4 / str_to_ov_type[normalized].size)

        return str_to_ov_type[normalized].size * total_elements

    # --- KVCache Interface Methods ---

    def get_k_cache(self) -> list[ov.Tensor]:
        """Returns the key cache tensors."""
        return self._k_cache

    def get_v_cache(self) -> list[ov.Tensor]:
        """Returns the value cache tensors."""
        return self._v_cache

