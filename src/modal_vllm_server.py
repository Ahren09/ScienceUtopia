"""Modal deployment of the Qwen3-32B vLLM OpenAI-compatible server.

Serves the IDENTICAL engine configuration used by all prior local runs
(vllm 0.12.0, TP=2, max-len 32768, gpu-mem 0.90, seed 42, qwen3 reasoning
parser, thinking mode driven per-request by the client), replicated across
up to 4 auto-scaling containers of 2x A100-80GB (8 GPUs max). Per-request
seeds make replica routing irrelevant to sampling.

Deploy:   modal deploy src/modal_vllm_server.py
Secret:   modal secret create science-utopia-vllm VLLM_API_KEY=<key>
Endpoint: https://<workspace>--science-utopia-vllm-serve.modal.run/v1

The local-A100 path is unaffected: clients keep selecting the backend purely
via --vllm_url; switching back requires no code change.
"""
import os
import subprocess

import modal

MODEL = "Qwen/Qwen3-32B"
PORT = 8000
VLLM_VERSION = "0.12.0"  # pinned to match all local runs

app = modal.App("science-utopia-vllm")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)

hf_cache = modal.Volume.from_name("science-utopia-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB:2",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("science-utopia-vllm")],
    timeout=24 * 60 * 60,
    scaledown_window=20 * 60,  # keep replicas warm 20 min after last request
    max_containers=10,         # up to 20 A100s when many runs execute in parallel
)
@modal.concurrent(max_inputs=32, target_inputs=24)  # spread load across replicas earlier
@modal.web_server(port=PORT, startup_timeout=3600)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--tensor-parallel-size", "2",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.90",
        "--port", str(PORT),
        "--seed", "42",
        "--reasoning-parser", "qwen3",
        "--api-key", os.environ["VLLM_API_KEY"],
    ]
    subprocess.Popen(cmd)
