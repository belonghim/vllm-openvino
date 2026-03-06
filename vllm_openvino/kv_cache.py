# SPDX-License-Identifier: Apache-2.0
"""An OpenVINO KV cache implementation for V1 KVCache interface."""
from typing import List, Optional, Tuple

import openvino as ov
import torch

import vllm_openvino.envs as envs
from vllm.attention.selector import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig)
from vllm.logger import init_logger
from vllm.platforms import current_platform

# Import V1 KVCache interface
from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)

str_to_ov_type = {
    "u8": ov.Type.u8,
    "i8": ov.Type.i8,
    "fp16": ov.Type.f16,
    "f16": ov.Type.f16,
    "bf16": ov.Type.bf16,
    "f32": ov.Type.f32,
    "fp32": ov.Type.f32,
    "dynamic": ov.Type.dynamic,
}

class OpenVINOCacheEngine:
    """Manages the KV cache for OpenVINO backend, implementing the V1 KVCache interface.

    This class is responsible for initializing and managing CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        key_cache_config: List[ov.PartialShape],
        value_cache_config: List[ov.PartialShape],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        ov_core: ov.Core,
        ov_device: str,
    ) -> None:
        assert device_config.device_type == "cpu"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.key_cache_config = key_cache_config
        self.value_cache_config = value_cache_config
        self.num_layers = len(self.value_cache_config)

        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for OpenVINO backend with a CPU target device, because we want
        # to reuse KV cache management in the scheduler.
        self.num_device_blocks = cache_config.num_gpu_blocks
        self.num_swap_blocks = cache_config.num_cpu_blocks

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        self.ov_cache_dtype = str_to_ov_type[self.cache_config.cache_dtype]

        # Initialize the cache.
        self.kv_cache: List[Tuple[ov.Tensor,
                                  ov.Tensor]] = self._allocate_kv_cache(
                                      self.num_device_blocks, ov_core,
                                      ov_device)

        # Initialize the swap.
        self.swap_cache: List[Tuple[ov.Tensor,
                                    ov.Tensor]] = self._allocate_swap_cache(
                                        self.num_swap_blocks, ov_device)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        ov_core: ov.Core,
        ov_device: str,
    ) -> List[Tuple[ov.Tensor, ov.Tensor]]:
        """Allocates KV cache."""
        kv_cache: List[Tuple[ov.Tensor, ov.Tensor]] = []

        for key_cache_pshape, value_cache_pshape in zip(self.key_cache_config, self.value_cache_config):
            key_cache_shape = key_cache_pshape
            value_cache_shape = value_cache_pshape
            key_cache_shape[0] = num_blocks
            value_cache_shape[0] = num_blocks
            key_cache_shape = key_cache_shape.to_shape()
            value_cache_shape = value_cache_shape.to_shape()

            if current_platform.is_openvino_cpu():
                key_blocks = ov.Tensor(self.ov_cache_dtype, key_cache_shape)
                value_blocks = ov.Tensor(self.ov_cache_dtype, value_cache_shape)
                kv_cache.append((key_blocks, value_blocks))
            else:
                remote_context = ov_core.get_default_context(ov_device)
                key_blocks = remote_context.create_tensor(self.ov_cache_dtype, key_cache_shape, {{}})
                value_blocks = remote_context.create_tensor(self.ov_cache_dtype, value_cache_shape, {{}})
                kv_cache.append((key_blocks, value_blocks))

        return kv_cache

    def _allocate_swap_cache(
        self,
        num_blocks: int,
        ov_device: str,
    ) -> List[Tuple[ov.Tensor, ov.Tensor]]:
        """Allocates swap cache."""
        swap_cache: List[Tuple[ov.Tensor, ov.Tensor]] = []

        if num_blocks == 0:
            return swap_cache

        assert not current_platform.is_openvino_cpu(), \
            "CPU device isn't supposed to have swap cache"

        for key_cache_pshape, value_cache_pshape in zip(self.key_cache_config, self.value_cache_config):
            key_cache_shape = key_cache_pshape
            value_cache_shape = value_cache_pshape
            key_cache_shape[0] = num_blocks
            value_cache_shape[0] = num_blocks

            key_blocks = ov.Tensor(self.ov_cache_dtype, key_cache_shape.to_shape())
            value_blocks = ov.Tensor(self.ov_cache_dtype, value_cache_shape.to_shape())
            swap_cache.append((key_blocks, value_blocks))

        return swap_cache

    def swap_in(self, src_to_dst: List[Tuple[int, int]]) -> None:
        for i in range(self.num_layers):
            for swap_tensor, kv_tensor in zip(self.swap_cache[i],
                                              self.kv_cache[i]):
                self.attn_backend.swap_blocks(swap_tensor, kv_tensor,
                                              src_to_dst)

    def swap_out(self, src_to_dst: List[Tuple[int, int]]) -> None:
        for i in range(self.num_layers):
            for swap_tensor, kv_tensor in zip(self.swap_cache[i],
                                              self.kv_cache[i]):
                self.attn_backend.swap_blocks(kv_tensor, swap_tensor,
                                              src_to_dst)

    def copy(self, src_to_dsts: List[Tuple[int, int]]) -> None:
        if (len(src_to_dsts) > 0):
            self.attn_backend.copy_blocks(self.kv_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_dtype: str,
        key_cache_config: List[ov.PartialShape],
        value_cache_config: List[ov.PartialShape],
    ) -> int:
        total_elements = 0
        for key_cache_shape, value_cache_shape in zip(key_cache_config, value_cache_config):
             total_elements += key_cache_shape[1].get_length() * key_cache_shape[2].get_length() * key_cache_shape[3].get_length()
             total_elements += value_cache_shape[1].get_length() * value_cache_shape[2].get_length() * value_cache_shape[3].get_length()
        return str_to_ov_type[cache_dtype].size * total_elements

    # --- KVCache Interface Methods ---

    def get_k_cache(self) -> List[ov.Tensor]:
        """Returns the key cache tensors."""
        return [tensor[0] for tensor in self.kv_cache]

    def get_v_cache(self) -> List[ov.Tensor]:
        """Returns the value cache tensors."""
        return [tensor[1] for tensor in self.kv_cache]

    def clone(self) -> 'OpenVINOCacheEngine':
        """Creates a deep copy of the KV cache.

        Note: This is a simplified clone. A full implementation might need to
        consider block allocation and copying semantics more deeply.
        """
        # This is a placeholder. A proper clone would involve allocating new
        # tensors and copying the data.
        # For now, we'll return a new instance with the same configuration.
        # A more robust implementation would copy the actual tensor data.
        new_cache_engine = OpenVINOCacheEngine(
            self.cache_config,
            self.key_cache_config,
            self.value_cache_config,
            self.model_config,
            self.parallel_config,
            DeviceConfig(device_type="cpu"), # Assuming CPU for OpenVINO
            ov.Core(), # This might need to be passed or managed differently
            envs.VLLM_OPENVINO_DEVICE # Assuming same device
        )
        # Copying the actual tensor data would be complex and might require
        # specific OpenVINO tensor operations.
        # For now, we'll just initialize it with the same number of blocks.
        # A more complete implementation would copy the contents of self.kv_cache.
        return new_cache_engine

    def grow(self, additional_blocks: int) -> None:
        """Grows the KV cache by allocating additional blocks.

        This method is complex as it involves reallocating tensors and copying
        existing data. For OpenVINO, this might require specific tensor
        management.
        """
        # This is a complex operation. For now, we'll raise a NotImplementedError.
        # A full implementation would need to: 
        # 1. Calculate new total blocks.
        # 2. Allocate new tensors with the increased size.
        # 3. Copy existing data from old tensors to new tensors.
        # 4. Update self.kv_cache.
        raise NotImplementedError("grow() is not yet implemented for OpenVINOCacheEngine.")

    def get_slot_kv_cache(self, slot: int) -> Tuple[ov.Tensor, ov.Tensor]:
        """Returns the KV cache tensors for a specific slot (sequence).

        This method assumes that the KV cache is structured such that each
        layer's cache can be indexed by slot. This might require a different
        internal representation or mapping.
        """
        # This method's implementation depends heavily on how slots are mapped
        # to blocks and how sequences are managed within the cache.
        # The current OpenVINOCacheEngine uses a block-based allocation.
        # A direct mapping to 'slot' might not be straightforward without
        # additional sequence management logic.
        raise NotImplementedError("get_slot_kv_cache() is not yet implemented for OpenVINOCacheEngine.")

    # The `copy` method is already present in the original class and seems to
    # align with the KVCache interface's `copy_blocks` functionality.
    # We will ensure it's correctly exposed.

    # The `swap_in` and `swap_out` methods are specific to OpenVINO's swap
    # mechanism and might not be directly part of the generic KVCache interface,
    # but they are kept for OpenVINO's internal use.



