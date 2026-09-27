"""Named Modal deployment configurations and their shared server construction."""

from dataclasses import dataclass
import os
import subprocess


@dataclass(frozen=True)
class ModalServerConfig:
    app_name: str
    cache_name: str
    secret_name: str
    gpu_count: int
    seed: int
    max_containers: int
    revision: str | None = None
    model: str = "Qwen/Qwen3-32B"
    port: int = 8000
    vllm_version: str = "0.12.0"
    timeout: int = 24 * 60 * 60
    scaledown_window: int = 20 * 60
    max_inputs: int = 32
    target_inputs: int = 24


STANDARD = ModalServerConfig(
    app_name="science-utopia-vllm",
    cache_name="science-utopia-hf-cache",
    secret_name="science-utopia-vllm",
    gpu_count=2,
    seed=42,
    max_containers=10,
)
RESOURCE_SIZE = ModalServerConfig(
    app_name="science-utopia-irxs-vllm",
    cache_name="science-utopia-irxs-hf-cache",
    secret_name="science-utopia-irxs-vllm",
    gpu_count=4,
    seed=8301,
    max_containers=1,
    revision="9216db5781bf21249d130ec9da846c4624c16137",
)


def server_command(config, api_key):
    command = ["vllm", "serve", config.model]
    if config.revision is not None:
        command.extend(["--revision", config.revision])
    command.extend(
        [
            "--tensor-parallel-size",
            str(config.gpu_count),
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.90",
            "--port",
            str(config.port),
            "--seed",
            str(config.seed),
            "--reasoning-parser",
            "qwen3",
            "--api-key",
            api_key,
        ]
    )
    return command


def deployment(config, function):
    """Declare one Modal app; processes start only inside its served function."""
    import modal

    app = modal.App(config.app_name)
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
        )
        .pip_install(f"vllm=={config.vllm_version}", "huggingface_hub[hf_transfer]")
        .env(
            {"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        )
    )
    cache = modal.Volume.from_name(config.cache_name, create_if_missing=True)

    serving = modal.web_server(port=config.port, startup_timeout=3600)(function)
    serving = modal.concurrent(
        max_inputs=config.max_inputs, target_inputs=config.target_inputs
    )(serving)
    serving = app.function(
        image=image,
        gpu=f"A100-80GB:{config.gpu_count}",
        volumes={"/root/.cache/huggingface": cache},
        secrets=[modal.Secret.from_name(config.secret_name)],
        timeout=config.timeout,
        scaledown_window=config.scaledown_window,
        max_containers=config.max_containers,
    )(serving)
    return app, serving


def start_server(config):
    """Start the configured server inside its Modal container."""
    subprocess.Popen(server_command(config, os.environ["VLLM_API_KEY"]))
