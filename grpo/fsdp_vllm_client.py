#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone FSDP-aware vLLM client for weight synchronization.

Communicates with vllm_trl_serve.py via:
  - HTTP for generation (/generate) and control messages
  - NCCL (via PyNcclCommunicator) for GPU-to-GPU weight transfer

NCCL Weight Sync Protocol:
  1. init_communicator:
     - Client POSTs to server → server picks host/port, fires collective_rpc
       in background, returns {host, port} immediately
     - Client creates StatelessProcessGroup connecting to host:port
     - TCP store rendezvous completes when all ranks join
     - Both sides create PyNcclCommunicator

  2. update_named_param:
     - Client POSTs param metadata → server fires collective_rpc in background,
       returns immediately
     - Client calls pynccl.broadcast(send, src=trainer_rank)
     - Workers (in collective_rpc) call pynccl.broadcast(recv, src=trainer_rank)
     - NCCL synchronizes both sides — client blocks until workers receive
     - This serializes updates: client can't send next param until current
       broadcast completes, which means server's collective_rpc also finished

  3. close_communicator:
     - Fire-and-forget cleanup

Usage:
    from fsdp_vllm_client import FSDPVLLMClient

    client = FSDPVLLMClient("http://host1:8000")
    client.init_communicator(device)
    client.update_model_params_streaming_fsdp(fsdp_model, trainable_only=True)
    response = client.generate(prompts, max_tokens=192, temperature=0.7)
    client.close_communicator()
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _ts() -> str:
    """Compact timestamp for recovery logging."""
    return time.strftime("%H:%M:%S", time.localtime())


class VLLMRecoveryNeeded(Exception):
    """Raised after vLLM connection is recovered but weights need resync.

    The caller (trainer step()) should:
      1. Broadcast this signal to all ranks
      2. Do weight resync with all ranks (summon_full_params is collective)
      3. Retry generation
    """
    pass


class FSDPVLLMClient:
    """vLLM client with FSDP streaming weight sync and TRL-format generation."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        connection_timeout: int = 240,
        max_retries: int = 10,
    ):
        self.base_url = (base_url or "http://localhost:8000").rstrip("/")
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self.session = requests.Session()

        self._pynccl_comm = None
        self._device = None
        self._tp_size = None

        # Track whether the first generate call has completed.
        # The first call may block for a long time if vLLM server's warmup
        # didn't fully cover all kernel variants (unlikely but possible).
        # Use extended timeout until first success, then switch to normal.
        self._first_generate_done = False

        # Only rank 0 does NCCL weight sync. All ranks can use HTTP generation.
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._is_main = (self._rank == 0)

        # Recovery attempt counter (persists across generate calls)
        self._recovery_count = 0

        self._wait_for_server()

    # ================================================================
    # Server health
    # ================================================================

    def _wait_for_server(self, label: str = "startup"):
        """Wait for /health to return 200. `label` is used in log messages."""
        deadline = time.time() + self.connection_timeout
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            remaining = max(0, deadline - time.time())
            try:
                resp = self.session.get(f"{self.base_url}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info(
                        "[%s] [%s] vLLM server healthy at %s (attempt %d)",
                        _ts(), label, self.base_url, attempt,
                    )
                    return
            except requests.ConnectionError:
                if attempt % 15 == 1:  # Log every ~30s
                    logger.info(
                        "[%s] [%s] waiting for vLLM... (attempt %d, %.0fs remaining)",
                        _ts(), label, attempt, remaining,
                    )
            except Exception as e:
                if attempt % 15 == 1:
                    logger.debug(
                        "[%s] [%s] health check error (attempt %d): %s",
                        _ts(), label, attempt, e,
                    )
            time.sleep(2.0)
        raise TimeoutError(
            f"[{label}] vLLM server at {self.base_url} not healthy "
            f"within {self.connection_timeout}s ({attempt} attempts)"
        )

    def _get_tp_size(self) -> int:
        if self._tp_size is not None:
            return self._tp_size
        try:
            resp = self.session.get(
                f"{self.base_url}/get_tensor_parallel_size", timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            # Handle both {"tensor_parallel_size": N} and plain int
            if isinstance(data, dict):
                self._tp_size = int(
                    data.get("tensor_parallel_size", data.get("world_size", 1))
                )
            else:
                self._tp_size = int(data)
            return self._tp_size
        except Exception:
            logger.warning("Could not get TP size, assuming 1")
            self._tp_size = 1
            return 1

    # ================================================================
    # NCCL Communicator
    # ================================================================

    def init_communicator(self, device: Optional[torch.device] = None, label: str = "init"):
        """
        Initialize NCCL communicator for weight sync.
        Only rank 0 of the trainer participates in the NCCL group.

        Protocol:
          1. POST /init_communicator → server returns {host, port, world_size}
             Server fires collective_rpc in background (workers create their
             StatelessProcessGroup and block on TCP store rendezvous).
          2. Client creates StatelessProcessGroup at same host:port.
             TCP store rendezvous completes when all ranks join.
          3. Both sides create PyNcclCommunicator.
        """
        if not self._is_main:
            logger.debug("Rank %d: skipping init_communicator (rank 0 only)", self._rank)
            return

        logger.info("[%s] [%s] init_communicator starting...", _ts(), label)

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = device

        tp_size = self._get_tp_size()
        world_size = tp_size + 1
        trainer_rank = tp_size  # last rank = trainer

        # Get CUDA device UUID for duplicate GPU detection
        client_device_uuid = ""
        if device is not None and device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(device)
                # UUID format varies by PyTorch version
                client_device_uuid = str(getattr(props, 'uuid', ''))
            except Exception:
                pass

        # Tell server to init (fire-and-forget on server side).
        # Server returns {host, port} where rank 0's TCP store will run.
        logger.info(
            "[%s] [%s] POST /init_communicator (tp=%d, world_size=%d)",
            _ts(), label, tp_size, world_size,
        )
        t0 = time.time()
        resp = self.session.post(
            f"{self.base_url}/init_communicator",
            json={
                "host": "", "port": 0, "world_size": world_size,
                "client_device_uuid": client_device_uuid,
            },
            timeout=30,
        )
        resp.raise_for_status()
        init_data = resp.json()
        logger.info(
            "[%s] [%s] POST /init_communicator OK in %.1fs: %s",
            _ts(), label, time.time() - t0, init_data,
        )

        host = init_data["host"]
        port = int(init_data["port"])
        server_world_size = int(init_data.get("world_size", world_size))

        logger.info(
            "[%s] [%s] Creating StatelessProcessGroup: host=%s, port=%d, "
            "rank=%d, world_size=%d",
            _ts(), label, host, port, trainer_rank, server_world_size,
        )

        # Create client-side process group.
        # This blocks until all ranks (workers + this client) have joined
        # the TCP store rendezvous. Workers join via the server's background
        # collective_rpc.
        t0 = time.time()
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=trainer_rank,
            world_size=server_world_size,
        )
        logger.info(
            "[%s] [%s] StatelessProcessGroup created in %.1fs",
            _ts(), label, time.time() - t0,
        )

        t0 = time.time()
        self._pynccl_comm = PyNcclCommunicator(pg, device=self._device)
        logger.info(
            "[%s] [%s] PyNcclCommunicator ready in %.1fs — rank=%d, world_size=%d, tp=%d",
            _ts(), label, time.time() - t0, trainer_rank, server_world_size, tp_size,
        )

    def close_communicator(self):
        """Close the NCCL communicator (rank 0 only)."""
        if not self._is_main:
            return
        if self._pynccl_comm is not None:
            try:
                self.session.post(
                    f"{self.base_url}/close_communicator", timeout=5
                )
            except Exception:
                pass  # Server may be dead — that's OK
            try:
                del self._pynccl_comm
            except Exception:
                pass
            self._pynccl_comm = None

    # ================================================================
    # Prefix cache reset
    # ================================================================

    def reset_prefix_cache(self):
        """Reset vLLM prefix cache (call after weight update).

        After model weights change, any cached KV blocks are stale and
        MUST be invalidated.  Without this, vLLM may hang or produce
        incorrect outputs.  See:
          - vllm-project/vllm#9744
          - huggingface/trl#3024
        """
        if not self._is_main:
            return
        try:
            resp = self.session.post(
                f"{self.base_url}/reset_prefix_cache", timeout=30,
            )
            resp.raise_for_status()
            logger.debug("Prefix cache reset OK")
        except Exception as e:
            logger.warning("Prefix cache reset failed (non-fatal): %s", e)

    # ================================================================
    # Weight Sync (per-param)
    # ================================================================

    def update_named_param(self, name: str, data: torch.Tensor):
        """
        Send a single parameter to vLLM workers via NCCL broadcast.
        Only rank 0 of the trainer performs this operation.

        Protocol:
          1. POST param metadata → server fires collective_rpc in background
             (returns immediately). Workers will call broadcast(recv).
          2. Client calls broadcast(send). NCCL synchronizes both sides.
          3. Client's broadcast returns only after workers have received,
             so it's safe to proceed to the next parameter.
        """
        if not self._is_main:
            return

        if self._pynccl_comm is None:
            raise RuntimeError("Call init_communicator() first")

        # Notify server (fire-and-forget on server side)
        resp = self.session.post(
            f"{self.base_url}/update_named_param",
            json={
                "name": name,
                "dtype": str(data.dtype),
                "shape": list(data.shape),
            },
            timeout=30,
        )
        resp.raise_for_status()

        # NCCL broadcast: trainer sends, workers receive.
        # src = world_size - 1 = trainer_rank.
        # This blocks until ALL ranks have called broadcast.
        data_gpu = data.to(self._device)
        self._pynccl_comm.broadcast(
            data_gpu, src=self._pynccl_comm.world_size - 1
        )

    def update_model_params(
        self, model: torch.nn.Module, trainable_only: bool = False,
    ):
        """Send model params to vLLM (non-FSDP).  Resets prefix cache after."""
        for name, param in model.named_parameters():
            if trainable_only and not param.requires_grad:
                continue
            clean = _clean_param_name(name)
            self.update_named_param(clean, param.data)
        self.reset_prefix_cache()

    # ================================================================
    # Weight Sync (FSDP streaming)
    # ================================================================

    def update_model_params_streaming_fsdp(
        self, fsdp_model: torch.nn.Module, trainable_only: bool = False,
    ):
        """
        Stream FSDP-sharded weights to vLLM, one FSDP unit at a time.

        ALL ranks must call this (FSDP.summon_full_params is collective),
        but only rank 0 performs the actual NCCL broadcast to vLLM.

        After all parameters are sent, the prefix cache is reset to
        invalidate stale KV blocks from the previous model weights.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if self._is_main and self._pynccl_comm is None:
            raise RuntimeError("Call init_communicator() first")

        rank = self._rank

        if isinstance(fsdp_model, FSDP):
            self._stream_fsdp_params(fsdp_model, trainable_only, rank)
        else:
            if rank == 0:
                self.update_model_params(fsdp_model, trainable_only)

    def _stream_fsdp_params(self, fsdp_model, trainable_only: bool, rank: int):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # Trigger FSDP lazy init on root before per-unit summon.
        # FSDP normally initializes _is_root and handle attrs during the first
        # forward pass.  If we call summon_full_params on child units before
        # that, _lazy_init fails because root hasn't set _is_root yet.
        # A no-op summon on root (recurse=False) triggers _lazy_init safely
        # without materializing the entire model.
        logger.debug("Triggering FSDP lazy init on root module...")
        with FSDP.summon_full_params(fsdp_model, writeback=False, recurse=False, rank0_only=True):
            pass
        logger.debug("FSDP lazy init complete.")

        # TRL-style sync (see trl GRPOTrainer._sync_fsdp1_params_to_vllm):
        # post-order traversal from the root, tracking each module's full
        # path within the model.  For every FSDP unit, materialize its own
        # (non-nested) parameters with summon_full_params(recurse=False,
        # writeback=False) and read them via named_parameters() with
        # recurse=True — in FSDP1 the flat parameter lives on the wrapped
        # module, so recurse=False yields nothing (a silent no-op sync).
        # Parameters owned by nested child FSDP units are sent when the
        # child itself is summoned (post-order: children first) and are
        # skipped in ancestors via the `visited` set of full names.
        def _has_trainable(unit):
            try:
                return any(p.requires_grad for p in unit.parameters())
            except Exception:
                return True

        visited = set()
        counters = {"sent": 0, "skipped_units": 0}

        def _sync_module(module, prefix: str):
            # Recurse into children first (post-order) so nested FSDP units
            # are fully materialized and sent before their ancestors.
            for child_name, child in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                _sync_module(child, child_prefix)

            if not isinstance(module, FSDP):
                return
            if trainable_only and not _has_trainable(module):
                counters["skipped_units"] += 1
                return

            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for name, param in module.named_parameters():
                    # Send name = module's path within the model + local name.
                    full_name = f"{prefix}.{name}" if prefix else name
                    full_name = _clean_param_name(full_name)
                    if full_name in visited:
                        continue  # owned by a nested FSDP child, already sent
                    visited.add(full_name)
                    if trainable_only and not param.requires_grad:
                        continue
                    if rank == 0:
                        self.update_named_param(full_name, param.data)
                        counters["sent"] += 1

        _sync_module(fsdp_model, "")

        # Invalidate stale prefix cache after weight update
        if rank == 0:
            self.reset_prefix_cache()
            msg = (
                f"FSDP streaming sync: sent {counters['sent']} params, "
                f"skipped {counters['skipped_units']} units"
            )
            logger.info(msg)
            # Also print: trainer scripts may not configure logging, and a
            # silent 0-param sync must be visible.
            print(f"[fsdp_vllm_client] {msg}", flush=True)
            if counters["sent"] == 0:
                print(
                    "[fsdp_vllm_client] WARNING: weight sync sent 0 parameters "
                    "— vLLM is still serving stale weights!",
                    flush=True,
                )

        if dist.is_initialized():
            dist.barrier()

    # ================================================================
    # Generation (HTTP → /generate)
    # ================================================================

    def generate(
        self,
        prompts: Union[str, List[str]],
        n: int = 1,
        max_tokens: int = 192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        min_p: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate via the TRL-format /generate endpoint.
        Returns dict with 'completion_ids' (list of token-ID sequences).

        On timeout (vLLM hang/crash), automatically recovers:
          1. Close stale NCCL communicator
          2. Wait for vLLM server to restart (auto-restart loop in shell)
          3. Re-establish NCCL communicator
          4. Raise VLLMRecoveryNeeded so caller can resync weights and retry
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        payload = {
            "prompts": prompts,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
        }
        for k, v in kwargs.items():
            if k not in payload:
                payload[k] = v

        url = f"{self.base_url}/generate"

        # --- First attempt ---
        gen_error = None
        try:
            timeout = self._get_generate_timeout()
            logger.info(
                "[%s] [generate] POST /generate: %d prompts, n=%d, timeout=%ds",
                _ts(), len(prompts), n, timeout,
            )
            t0 = time.time()
            resp = self.session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            elapsed = time.time() - t0
            logger.info(
                "[%s] [generate] OK in %.1fs (%d prompts × n=%d)",
                _ts(), elapsed, len(prompts), n,
            )
            self._first_generate_done = True
            return resp.json()
        except Exception as first_error:
            gen_error = first_error
            elapsed = time.time() - t0 if 't0' in dir() else 0
            if not self._first_generate_done:
                logger.warning(
                    "[%s] [generate] First generate failed after %.1fs: %s — using retry logic",
                    _ts(), elapsed, gen_error,
                )
                return self._generate_with_retries(url, payload)

            self._recovery_count += 1
            logger.warning(
                "[%s] [RECOVERY #%d] Generate failed after %.1fs: %s",
                _ts(), self._recovery_count, elapsed, gen_error,
            )

        # --- Recovery: reconnect to restarted vLLM ---
        recovery_label = f"recovery#{self._recovery_count}"
        logger.info(
            "[%s] [%s] ====== RECOVERY START ======",
            _ts(), recovery_label,
        )

        recovery_err = None
        recovery_t0 = time.time()
        try:
            self._recover_connection(label=recovery_label)
        except Exception as e:
            logger.error(
                "[%s] [%s] _recover_connection FAILED after %.1fs: %s",
                _ts(), recovery_label, time.time() - recovery_t0, e,
            )
            recovery_err = e

        recovery_elapsed = time.time() - recovery_t0
        if recovery_err is None:
            logger.info(
                "[%s] [%s] ====== RECOVERY SUCCESS (%.1fs) ======",
                _ts(), recovery_label, recovery_elapsed,
            )
        else:
            logger.error(
                "[%s] [%s] ====== RECOVERY FAILED (%.1fs) ======",
                _ts(), recovery_label, recovery_elapsed,
            )

        raise VLLMRecoveryNeeded(
            f"vLLM recovery {'succeeded' if recovery_err is None else 'FAILED'} "
            f"in {recovery_elapsed:.1f}s after: {gen_error}. "
            f"recovery_err={recovery_err}. "
            f"Caller must resync weights and retry generation."
        )

    def _get_generate_timeout(self) -> int:
        """Return appropriate timeout for generate calls."""
        if self._first_generate_done:
            return 600  # 10 min — covers max_num_seqs=64 batch splitting
        else:
            logger.info(
                "[%s] First generate call — using extended timeout (7200s) "
                "as safety net for JIT compilation", _ts(),
            )
            return 7200  # 2h — first call may trigger JIT

    def _generate_with_retries(self, url: str, payload: dict) -> Dict[str, Any]:
        """Retry logic for the very first generate (JIT warmup)."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "[%s] [first-gen] Attempt %d/%d (timeout=7200s)",
                    _ts(), attempt, self.max_retries,
                )
                t0 = time.time()
                resp = self.session.post(url, json=payload, timeout=7200)
                resp.raise_for_status()
                logger.info(
                    "[%s] [first-gen] OK on attempt %d in %.1fs",
                    _ts(), attempt, time.time() - t0,
                )
                self._first_generate_done = True
                return resp.json()
            except Exception as e:
                last_error = e
                elapsed = time.time() - t0
                if attempt < self.max_retries:
                    logger.warning(
                        "[%s] [first-gen] Attempt %d/%d failed after %.1fs: %s",
                        _ts(), attempt, self.max_retries, elapsed, e,
                    )
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"First generate failed after {self.max_retries} attempts: {last_error}"
        )

    def _recover_connection(
        self, server_wait_timeout: int = 1800, label: str = "recovery",
    ):
        """
        Recover from vLLM crash: force-exit the hung server, wait for the
        shell watchdog to auto-restart it (model reload + warmup), then
        re-establish the NCCL communicator.

        The auto-restart loop (while-true watchdog in the launcher shell
        script, mp executor backend) handles:
          1. pkill leftover vllm_trl_serve / multiprocessing workers
          2. Clean /dev/shm/vllm_* shared memory
          3. Relaunch vllm_trl_serve.py (model load + warmup)

        This can take 5-15 minutes depending on JIT cache state.
        Caller is responsible for weight resync after this returns.
        """
        if not self._is_main:
            return

        step_t0 = time.time()

        # ---- Step 1: Kill hung server ----
        logger.info("[%s] [%s] Step 1/4: Sending force_exit to vLLM...", _ts(), label)
        try:
            resp = self.session.post(
                f"{self.base_url}/force_exit", timeout=5,
            )
            logger.info(
                "[%s] [%s] Step 1/4: force_exit response: %s (%.1fs)",
                _ts(), label, resp.status_code, time.time() - step_t0,
            )
            time.sleep(5)  # Give it a moment to SIGKILL
        except requests.ConnectionError:
            logger.info(
                "[%s] [%s] Step 1/4: force_exit ConnectionError — server already dead (%.1fs)",
                _ts(), label, time.time() - step_t0,
            )
        except requests.Timeout:
            logger.info(
                "[%s] [%s] Step 1/4: force_exit Timeout — server hung, waiting for auto-restart (%.1fs)",
                _ts(), label, time.time() - step_t0,
            )
        except Exception as e:
            logger.info(
                "[%s] [%s] Step 1/4: force_exit error: %s (%.1fs)",
                _ts(), label, e, time.time() - step_t0,
            )

        # ---- Step 2: Close stale NCCL comm ----
        step_t0 = time.time()
        logger.info("[%s] [%s] Step 2/4: Closing stale NCCL communicator...", _ts(), label)
        if self._pynccl_comm is not None:
            try:
                del self._pynccl_comm
                logger.info("[%s] [%s] Step 2/4: NCCL comm deleted (%.1fs)", _ts(), label, time.time() - step_t0)
            except Exception as e:
                logger.warning("[%s] [%s] Step 2/4: NCCL comm delete error: %s", _ts(), label, e)
            self._pynccl_comm = None
        else:
            logger.info("[%s] [%s] Step 2/4: No NCCL comm to close", _ts(), label)

        # Reset HTTP session (stale connections)
        self.session.close()
        self.session = requests.Session()
        logger.info("[%s] [%s] Step 2/4: HTTP session reset (%.1fs)", _ts(), label, time.time() - step_t0)

        # ---- Step 3: Wait for vLLM auto-restart ----
        step_t0 = time.time()
        logger.info(
            "[%s] [%s] Step 3/4: Waiting for vLLM restart (timeout=%ds)...",
            _ts(), label, server_wait_timeout,
        )
        old_timeout = self.connection_timeout
        self.connection_timeout = server_wait_timeout
        try:
            self._wait_for_server(label=f"{label}/wait-restart")
            logger.info(
                "[%s] [%s] Step 3/4: vLLM server is back! (%.1fs)",
                _ts(), label, time.time() - step_t0,
            )
        except TimeoutError as e:
            logger.error(
                "[%s] [%s] Step 3/4: TIMEOUT waiting for vLLM restart (%.1fs): %s",
                _ts(), label, time.time() - step_t0, e,
            )
            raise
        finally:
            self.connection_timeout = old_timeout

        # ---- Step 3.5: Verify server is truly ready ----
        step_t0 = time.time()
        logger.info("[%s] [%s] Step 3.5/4: Verifying server stability...", _ts(), label)
        for verify_attempt in range(1, 6):
            try:
                resp = self.session.post(
                    f"{self.base_url}/generate",
                    json={"prompts": ["test"], "max_tokens": 1, "temperature": 0.0},
                    timeout=120,
                )
                resp.raise_for_status()
                logger.info(
                    "[%s] [%s] Step 3.5/4: Verify generate OK on attempt %d (%.1fs)",
                    _ts(), label, verify_attempt, time.time() - step_t0,
                )
                break
            except Exception as e:
                logger.warning(
                    "[%s] [%s] Step 3.5/4: Verify attempt %d/5 failed: %s",
                    _ts(), label, verify_attempt, e,
                )
                if verify_attempt == 5:
                    logger.error(
                        "[%s] [%s] Step 3.5/4: Server not stable after 5 attempts (%.1fs)",
                        _ts(), label, time.time() - step_t0,
                    )
                    raise RuntimeError(
                        f"vLLM server /health OK but /generate failed 5 times"
                    ) from e
                time.sleep(10)

        # ---- Step 4: Re-establish NCCL communicator ----
        step_t0 = time.time()
        logger.info("[%s] [%s] Step 4/4: Re-establishing NCCL communicator...", _ts(), label)
        self._tp_size = None  # Re-query from restarted server
        try:
            self.init_communicator(self._device, label=f"{label}/reconnect")
            logger.info(
                "[%s] [%s] Step 4/4: NCCL communicator re-established (%.1fs)",
                _ts(), label, time.time() - step_t0,
            )
        except Exception as e:
            logger.error(
                "[%s] [%s] Step 4/4: NCCL reconnect FAILED (%.1fs): %s",
                _ts(), label, time.time() - step_t0, e,
            )
            raise

        logger.info("[%s] [%s] vLLM connection recovered successfully", _ts(), label)


# ================================================================
# Utilities
# ================================================================

def _clean_param_name(name: str) -> str:
    """Remove FSDP wrapper prefixes from parameter names."""
    name = name.replace("_fsdp_wrapped_module.", "")
    name = name.replace("_checkpoint_wrapped_module.", "")
    name = name.replace("module.", "")
    return name


VLLMClient = FSDPVLLMClient
