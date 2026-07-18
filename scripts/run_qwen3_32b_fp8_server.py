#!/usr/bin/env python
"""Launch a single-GPU Qwen3-32B FP8 vLLM server (one GPU, one port).

First used by the network-proximity B-A confirmatory replay (GPU 2, port 8035,
the defaults); the scale-expansion experiment launches several of these in
parallel (e.g. GPUs 1-5 on ports 8031-8035) and assigns runs round-robin.
FP8 (~34GB weights) fits one 80GB A100 with room for KV cache, so concurrent
simulation/replay clients can stack against it. Same pinned model revision as
the paper's bf16 study; an FP8-vs-bf16 concordance check on the smoke set
(against the bf16 server on port 8036) justifies the precision choice.

Run from the repo root:
    python scripts/run_qwen3_32b_fp8_server.py                    # GPU 2, port 8035
    python scripts/run_qwen3_32b_fp8_server.py --gpu 1 --port 8031
Logs to outputs/logs/qwen3_32b_fp8_gpu<G>_port<P>/server.log; pid written next to it.
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = 'Qwen/Qwen3-32B'
REVISION = '9216db5781bf21249d130ec9da846c4624c16137'  # exact paper revision

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--gpu', default='2', help='CUDA device index for this server')
parser.add_argument('--port', default='8035')
args = parser.parse_args()
GPU, PORT = str(args.gpu), str(args.port)
LOG_DIR = os.path.join(REPO_ROOT, 'outputs', 'logs', f'qwen3_32b_fp8_gpu{GPU}_port{PORT}')
os.makedirs(LOG_DIR, exist_ok=True)

# Use the vllm that belongs to the interpreter running this script (a pinned venv overlay
# such as /workingdir/yjin328/venvs/scienceutopia_tf4573 keeps vllm 0.12.0 / torch 2.9.0+cu128
# even after the shared conda env was upgraded to an incompatible stack on 2026-09-08).
VLLM_BIN = os.path.join(os.path.dirname(sys.executable), 'vllm')
if not os.path.exists(VLLM_BIN):
    VLLM_BIN = 'vllm'
cmd = [
    VLLM_BIN, 'serve', MODEL,
    '--revision', REVISION,
    '--quantization', 'fp8',          # dynamic FP8 of the bf16 checkpoint (~34GB weights)
    '--tensor-parallel-size', '1',
    '--max-model-len', '32768',        # matches the frozen source config
    '--gpu-memory-utilization', '0.75',  # ~60GB; leaves headroom on the 80GB card
    '--port', PORT,
    '--seed', '42',
    '--reasoning-parser', 'qwen3',
]

env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = GPU
env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# Resolve libstdc++ from the conda env, not the (older) system one: libicu 78 in the env
# needs CXXABI_1.3.15, which /lib/x86_64-linux-gnu/libstdc++.so.6 lacks (server crash on
# 2026-09-10 at `import sqlite3` inside vllm.v1.structured_output).
env['LD_LIBRARY_PATH'] = os.path.join(sys.base_prefix, 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')

log_path = os.path.join(LOG_DIR, 'server.log')
print(f'[launcher] GPU{GPU} port {PORT} FP8 {MODEL}@{REVISION[:12]}')
print(f'[launcher] command: CUDA_VISIBLE_DEVICES={GPU} ' + ' '.join(cmd))
print(f'[launcher] logging to {log_path}')
with open(log_path, 'w') as logf:
    logf.write(f'# launched {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    logf.write('# CUDA_VISIBLE_DEVICES=%s %s\n' % (GPU, ' '.join(cmd)))
    logf.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT)
    print(f'[launcher] server pid {proc.pid}')
    with open(os.path.join(LOG_DIR, 'server.pid'), 'w') as f:
        f.write(str(proc.pid) + '\n')
    sys.exit(0)
