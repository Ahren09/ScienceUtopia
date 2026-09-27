"""Launch a named local vLLM profile with its established flags."""

import argparse
import os
import subprocess
import sys

PROFILES = {
    "_qwen3-8b": {
        "devices": "2",
        "arguments": [
            "--model",
            "Qwen/Qwen3-8B",
            "--gpu-memory-utilization",
            "0.95",
            "--max-model-len",
            "32768",
            "--tensor-parallel-size",
            "1",
            "--port",
            "8008",
        ],
    },
    "_qwen3-0.6b": {
        "devices": "5",
        "arguments": [
            "--model",
            "Qwen/Qwen3-0.6B",
            "--gpu-memory-utilization",
            "0.95",
            "--max-model-len",
            "16384",
        ],
    },
    "_qwen3-1.7b": {
        "devices": "3",
        "arguments": [
            "--model",
            "Qwen/Qwen3-1.7B",
            "--gpu-memory-utilization",
            "0.95",
            "--max-model-len",
            "32768",
            "--tensor-parallel-size",
            "1",
            "--reasoning-parser",
            "qwen3",
            "--port",
            "8017",
        ],
    },
    "_gemma3-27b": {
        "devices": "0,1,4,5",
        "arguments": [
            "--model",
            "google/gemma-3-27b-it",
            "--gpu-memory-utilization",
            "0.95",
            "--max-model-len",
            "16384",
            "--tensor-parallel-size",
            "2",
            "--port",
            "8027",
        ],
    },
    "_qwen3-8b_cot": {
        "devices": "4,5",
        "arguments": [
            "--model",
            "Qwen/Qwen3-8B",
            "--gpu-memory-utilization",
            "0.95",
            "--max-model-len",
            "32768",
            "--tensor-parallel-size",
            "2",
            "--reasoning-parser",
            "qwen3",
            "--port",
            "8000",
        ],
    },
    "_qwen3-32b": {
        "devices": "4,5",
        "arguments": [
            "--model",
            "Qwen/Qwen3-32B",
            "--gpu-memory-utilization",
            "0.95",
            "--max-model-len",
            "32768",
            "--tensor-parallel-size",
            "2",
            "--reasoning-parser",
            "qwen3",
            "--port",
            "8032",
        ],
    },
}


def server_command(profile, python=sys.executable):
    return [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        *PROFILES[profile]["arguments"],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    command = server_command(args.profile)
    if args.dry_run:
        import shlex

        print(shlex.join(command))
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=PROFILES[args.profile]["devices"])
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
