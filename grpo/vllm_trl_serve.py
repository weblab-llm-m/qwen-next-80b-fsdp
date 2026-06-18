#!/usr/bin/env python3
"""
Custom vLLM server with TRL WeightSyncWorkerExtension HTTP routes.

Bridges vLLM's LLM() offline API with the HTTP endpoints expected by the
GRPO trainer for weight synchronization (NCCL GPU-to-GPU transfer via the
triton_worker_extension worker extension).

CRITICAL DESIGN: init_communicator / update_named_param / close_communicator
use fire-and-forget for collective_rpc. The NCCL operations inside
collective_rpc are BLOCKING — they wait for ALL ranks (including the
trainer client) to participate. If the HTTP handler blocked on
collective_rpc, the client couldn't proceed to its NCCL call, causing
deadlock. Fire-and-forget lets the HTTP response return immediately so
the client can join the NCCL collective.

Routes:
  GET  /health                    - Health check
  GET  /get_world_size            - TP size for weight sync group
  GET  /get_tensor_parallel_size  - Alias
  POST /init_communicator         - Returns {host, port} for NCCL group
  POST /update_named_param        - Fire-and-forget → workers receive via NCCL
  POST /close_communicator        - Fire-and-forget
  POST /generate                  - Synchronous text generation
  POST /reset_prefix_cache        - Clear prefix cache
  POST /force_exit                - SIGKILL self so the shell watchdog restarts

Usage:
  python vllm_trl_serve.py \\
    --model /path/to/model \\
    --tensor-parallel-size 8 \\
    --distributed-executor-backend mp \\
    --port 8000 \\
    --trust-remote-code
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch
import uvicorn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IMPORTANT: Make this directory importable in *all* subprocesses / Ray actors.
# - worker_extension_cls is imported by name.
# - In Ray multi-node, working_dir ships this folder to workers.
# ---------------------------------------------------------------------------
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
os.environ["PYTHONPATH"] = _THIS_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

# Fail fast if the extension module isn't importable here (it must be for workers too).
try:
    import importlib
    importlib.import_module("triton_worker_extension")
except Exception as e:
    raise RuntimeError(
        "Failed to import 'triton_worker_extension'. "
        "Put triton_worker_extension.py in the SAME directory as vllm_trl_serve.py "
        "and ensure this directory is on PYTHONPATH."
    ) from e

# --- Triton allocator fix (server process only; workers/enginecore set the
# allocator themselves in triton_worker_extension) ---
try:
    import triton
    try:
        import triton.runtime.driver as _triton_driver
        triton.set_allocator(_triton_driver.CudaAllocator())
    except Exception:
        def _alloc_fn(size: int, alignment: int, stream: int | None):
            return torch.empty((size,), device="cuda", dtype=torch.int8)
        triton.set_allocator(_alloc_fn)
except Exception:
    pass

from fastapi import FastAPI
from pydantic import BaseModel

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# GDN (Gated Delta Net) FlashInfer JIT compilation can take long on first run.
os.environ.setdefault("RAY_CGRAPH_get_timeout", "7200")        # 2 hours
os.environ.setdefault("RAY_CGRAPH_teardown_timeout", "120")

from vllm import LLM, SamplingParams

# get_ip / get_open_port moved to vllm.utils.network_utils in vLLM >=0.14
import socket as _socket
try:
    from vllm.utils import get_ip, get_open_port
except ImportError:
    try:
        from vllm.utils.network_utils import get_open_port
    except ImportError:
        def get_open_port() -> int:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]
    try:
        from vllm.utils.network_utils import get_ip
    except ImportError:
        def get_ip() -> str:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
            except Exception:
                return "127.0.0.1"
            finally:
                s.close()

# GuidedDecodingParams was removed from vllm.sampling_params in v0.12.0
try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None

# Dedicated thread pool for fire-and-forget collective_rpc calls.
# Keep single-thread to serialize updates and avoid overlap/races.
_rpc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rpc")


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM + TRL weight sync server")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=64,
                        help="Max concurrent sequences in scheduler.")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--language-model-only", action="store_true",
                        help="Skip vision encoder (maps to limit_mm_per_prompt)")
    parser.add_argument("--distributed-executor-backend", type=str, default="ray",
                        choices=["ray", "mp", "uni"])
    parser.add_argument("--kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip warmup generate (not recommended for GDN models)")
    return parser.parse_args()


def warmup_generate(llm: LLM):
    logger.info("=" * 60)
    logger.info("WARMUP: Starting warmup generation to trigger JIT compilation")
    logger.info("=" * 60)

    warmup_prompt = "Hello"
    warmup_params = SamplingParams(max_tokens=1, temperature=0.0)

    t0 = time.time()
    outputs = llm.generate(prompts=[warmup_prompt], sampling_params=warmup_params)
    elapsed = time.time() - t0
    logger.info("WARMUP: Complete in %.1fs (%.1f min)", elapsed, elapsed / 60)
    if outputs and outputs[0].outputs:
        logger.info("WARMUP: Output token: %r", outputs[0].outputs[0].text[:50])


def _ray_runtime_env_for_this_dir():
    # Ship *this directory* to Ray workers, and ensure PYTHONPATH includes it.
    return {
        "working_dir": _THIS_DIR,
        "env_vars": {
            "PYTHONPATH": os.environ["PYTHONPATH"],
        },
    }


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("vLLM + TRL Weight Sync Server")
    logger.info("=" * 60)
    logger.info("  Model: %s", args.model)
    logger.info("  TP: %s", args.tensor_parallel_size)
    logger.info("  Backend: %s", args.distributed_executor_backend)
    logger.info("  Script dir: %s", _THIS_DIR)
    logger.info("  PYTHONPATH: %s", os.environ.get("PYTHONPATH", ""))

    llm_kwargs = dict(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        kv_cache_dtype=args.kv_cache_dtype,
        trust_remote_code=args.trust_remote_code,
        distributed_executor_backend=args.distributed_executor_backend,
        # worker_extension_cls must be importable by name on every worker.
        worker_extension_cls="triton_worker_extension.TritonAllocatorWeightSyncWorkerExtension",
    )

    if args.enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.language_model_only:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    # Key fix for Ray multi-node: ship our worker extension via working_dir.
    if args.distributed_executor_backend == "ray":
        llm_kwargs["ray_runtime_env"] = _ray_runtime_env_for_this_dir()

    llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}
    logger.info("  LLM kwargs keys: %s", sorted(llm_kwargs.keys()))

    logger.info("Creating LLM engine...")
    llm = LLM(**llm_kwargs)
    logger.info("LLM engine created successfully")

    if not args.skip_warmup:
        warmup_generate(llm)
    else:
        logger.info("WARMUP: Skipped (--skip-warmup)")

    nccl_host = os.environ.get("VLLM_HOST_IP", get_ip())
    logger.info("  NCCL master host: %s", nccl_host)

    app = FastAPI(title="vLLM + TRL Weight Sync Server")

    @app.get("/health")
    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size")
    @app.get("/get_world_size/")
    async def get_world_size():
        return {"world_size": args.tensor_parallel_size}

    @app.get("/get_tensor_parallel_size")
    @app.get("/get_tensor_parallel_size/")
    async def get_tensor_parallel_size():
        return {"tensor_parallel_size": args.tensor_parallel_size}

    class InitCommunicatorRequest(BaseModel):
        host: str = ""
        port: int = 0
        world_size: int = 0
        client_device_uuid: str = ""

    @app.post("/init_communicator")
    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        world_size = args.tensor_parallel_size + 1
        port = get_open_port()
        client_device_uuid = request.client_device_uuid

        logger.info(
            "init_communicator: host=%s port=%s world_size=%s client_device_uuid=%s",
            nccl_host, port, world_size, client_device_uuid
        )

        def _init():
            try:
                llm.collective_rpc(
                    method="init_communicator",
                    args=(nccl_host, port, world_size, client_device_uuid),
                )
                logger.info("init_communicator collective_rpc completed")
            except Exception as e:
                logger.exception("init_communicator collective_rpc failed: %s", e)

        _rpc_executor.submit(_init)
        return {"host": nccl_host, "port": port, "world_size": world_size}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param")
    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        # Robust dtype parse (accept "float16", "torch.float16", etc.)
        dt_name = request.dtype.split(".")[-1]
        if not hasattr(torch, dt_name):
            return {"error": f"Unknown dtype: {request.dtype}"}
        dtype = getattr(torch, dt_name)

        def _update():
            try:
                llm.collective_rpc(
                    method="update_named_param",
                    args=(request.name, dtype, tuple(request.shape)),
                )
            except Exception as e:
                logger.exception("update_named_param failed for %s: %s", request.name, e)

        _rpc_executor.submit(_update)
        return {"message": f"Update started for {request.name}"}

    @app.post("/close_communicator")
    @app.post("/close_communicator/")
    async def close_communicator():
        def _close():
            try:
                llm.collective_rpc(method="close_communicator")
                logger.info("close_communicator completed")
            except Exception as e:
                logger.exception("close_communicator failed: %s", e)

        _rpc_executor.submit(_close)
        return {"message": "Close initiated"}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate", response_model=GenerateResponse)
    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        if request.guided_decoding_regex is not None and GuidedDecodingParams is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        sp_kwargs = dict(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
        )
        if guided_decoding is not None:
            sp_kwargs["guided_decoding"] = guided_decoding

        sampling_params = SamplingParams(**sp_kwargs)

        def _generate():
            return llm.generate(prompts=request.prompts, sampling_params=sampling_params)

        all_outputs = await asyncio.to_thread(_generate)

        completion_ids = [
            list(output.token_ids)
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        return {"completion_ids": completion_ids}

    @app.post("/reset_prefix_cache")
    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        success = await asyncio.to_thread(llm.reset_prefix_cache)
        return {"message": f"Prefix cache reset: {success}"}

    @app.post("/force_exit")
    @app.post("/force_exit/")
    async def force_exit():
        import signal
        logger.warning("force_exit requested — SIGKILL in 0.5s for auto-restart")
        loop = asyncio.get_running_loop()
        loop.call_later(0.5, os.kill, os.getpid(), signal.SIGKILL)
        return {"message": "Shutting down"}

    logger.info("Starting server on %s:%s", args.host, args.port)
    uvicorn.run(
        app, host=args.host, port=args.port,
        log_level=args.log_level,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
