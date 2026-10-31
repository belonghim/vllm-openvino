from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM


class Lfm3ForCausalLM(Lfm2ForCausalLM):

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        hf_config = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=hf_config.hidden_size,
            conv_kernel=hf_config.conv_L_cache,
        )
