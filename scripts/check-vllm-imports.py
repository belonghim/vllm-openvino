# Fails if a vLLM symbol the plugin depends on no longer resolves: every
# `from vllm... import` (optional imports in try blocks excluded), the private
# model methods the runner borrows, and the scheduler/worker dataclass fields
# it reads or fills. Run from the repo root inside the target vLLM environment.
import ast, dataclasses, importlib, pathlib, sys

bad = []
for f in pathlib.Path("vllm_openvino").rglob("*.py"):
    tree = ast.parse(f.read_text())
    optional = {id(n) for t in ast.walk(tree) if isinstance(t, ast.Try)
                for n in ast.walk(t)}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.ImportFrom) and n.module
                and n.module.split(".")[0] == "vllm") or id(n) in optional:
            continue
        try:
            m = importlib.import_module(n.module)
        except Exception as e:
            bad.append(f"{f}: {n.module}: {e!r}")
            continue
        for a in n.names:
            if not hasattr(m, a.name):
                try:
                    importlib.import_module(f"{n.module}.{a.name}")
                except Exception:
                    bad.append(f"{f}: {n.module}.{a.name}")

ATTRS = {
    "vllm.model_executor.models.minicpmv4_7:MiniCPMV4_7ForConditionalGeneration":
        ["_init_canvas_mrope", "get_mrope_input_positions"],
    "vllm.model_executor.models.qwen3_vl:Qwen3VLForConditionalGeneration":
        ["_get_mrope_input_positions"],
    "vllm.v1.outputs:ModelRunnerOutput":
        ["req_ids", "sampled_token_ids", "logprobs", "prompt_logprobs_dict",
         "pooler_output"],
    "vllm.v1.outputs:DraftTokenIds": ["req_ids", "draft_token_ids"],
    "vllm.v1.core.sched.output:SchedulerOutput":
        ["scheduled_new_reqs", "scheduled_cached_reqs", "num_scheduled_tokens",
         "total_num_scheduled_tokens", "scheduled_spec_decode_tokens",
         "scheduled_encoder_inputs", "finished_req_ids",
         "free_encoder_mm_hashes", "preempted_req_ids",
         "new_block_ids_to_zero", "num_spec_tokens_to_schedule"],
    "vllm.v1.core.sched.output:CachedRequestData":
        ["req_ids", "new_block_ids", "num_computed_tokens"],
    "vllm.v1.core.sched.output:NewRequestData":
        ["req_id", "prompt_token_ids", "prefill_token_ids", "mm_features",
         "sampling_params", "pooling_params", "block_ids",
         "num_computed_tokens", "lora_request"],
}
for target, names in ATTRS.items():
    module, cls_name = target.split(":")
    try:
        cls = getattr(importlib.import_module(module), cls_name)
    except Exception as e:
        bad.append(f"{target}: {e!r}")
        continue
    fields = set(getattr(cls, "__struct_fields__", ()))
    if dataclasses.is_dataclass(cls):
        fields |= {field.name for field in dataclasses.fields(cls)}
    bad += [f"{target}.{name}" for name in names
            if name not in fields and not hasattr(cls, name)]

print("\n".join(bad) or "all vLLM imports, borrowed methods and fields resolve")
sys.exit(1 if bad else 0)
