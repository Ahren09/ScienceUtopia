"""One prospective TP1 runtime contract, shared by controller and driver.

This module is stdlib-only and starts no processes. The plan label and runtime
identifier name the same fixed profile; they are not arbitrary configuration
overrides. The accessor returns a fresh dict so callers cannot change the
contract for another job. TP2's existing contract remains in its original code.
"""

from types import MappingProxyType


TP1_PROFILE_NAME = "qwen3_32b_bf16_tp1_v1"
TP1_PLAN_PROFILE = "propensity_tp1_v1"
TP1_SERVER_SPEC = MappingProxyType(
    {
        "runtime_profile": TP1_PROFILE_NAME,
        "model": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.94,
        "enforce_eager": True,
        "max_num_batched_tokens": 2048,
        "enable_chunked_prefill": True,
        "seed": 42,
        "request_seed": 42,
        "request_seed_base": 42,
        "request_seed_policy": "derive_seed(run_seed, phase_context, item_index, retry)",
        "reasoning_parser": "qwen3",
        "vllm_version": "0.12.0",
        "transformers_version": "4.57.3",
        "torch_version": "2.9.0+cu128",
    }
)


def tp1_server_spec():
    return dict(TP1_SERVER_SPEC)
