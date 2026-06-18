#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-parameter RL with GRPO-style objective and LLM-as-judge reward.
Adapted for Qwen3-Next-80B-A3B (48 layers, 512 MoE experts, hybrid attention, BF16).

[v2] Judge reward changed to benchmark-style 0-9 scoring with original_truth.

Architecture (from config.json):
  - 48 layers, hidden_size=2048, head_dim=256
  - 512 routed experts (10 active/tok) + 1 shared expert per layer
  - Hybrid attention: full_attention_interval=4
    → layers 0,4,8,...,44 = Full Attention (12 layers)
    → others = Gated DeltaNet (36 layers)
  - Multi-Token Prediction (MTP) heads (ignored for training)
  - BF16 native, vocab_size=151936
  - Total 80B params, 3B active per token

Node layout (1+4 nodes):
  vLLM server : 1 node, 8 GPU, TP=8 (mp backend)
  Trainer     : 4 nodes, 32 GPU, FSDP FULL_SHARD

Memory estimate (32 GPU, full-param BF16):
  Weights     : ~160GB / 32 = 5GB/GPU
  Adam states : ~640GB / 32 = 20GB/GPU
  Gradients   : ~160GB / 32 = 5GB/GPU
  Total       : ~30GB/GPU  (H100 80GB → comfortable)
"""

import os
import re
import json
import math
import random
import argparse
import hashlib
import time
import datetime
import threading
from typing import List, Tuple, Optional, Any
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Standalone FSDP vLLM client
try:
    from fsdp_vllm_client import FSDPVLLMClient as VLLMClient
    from fsdp_vllm_client import VLLMRecoveryNeeded
except ImportError:
    VLLMRecoveryNeeded = None
    try:
        from trl.generation.vllm_client import VLLMClient
    except ImportError:
        try:
            from trl.extras.vllm_client import VLLMClient
        except ImportError:
            VLLMClient = None

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

def extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [extract_text_from_content(x) for x in content]
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        inner = content.get("content")
        if isinstance(inner, str):
            return inner
        if isinstance(inner, (list, dict)):
            return extract_text_from_content(inner)
        parts = [v for v in content.values() if isinstance(v, str)]
        return "\n".join(parts)
    return ""


class ChatMLJsonl(IterableDataset):
    def __init__(
        self,
        path: str,
        rank: int = 0,
        world_size: int = 1,
        tokenizer=None,
        max_prompt_tokens: Optional[int] = None,
        filter_overlong: bool = True,
        log_drops: bool = False,
        drop_log_every: int = 2000,
    ):
        self.path = path
        self.rank = rank
        self.world_size = world_size

        # for overlong prompt filtering
        self.tok = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens) if max_prompt_tokens is not None else None
        self.filter_overlong = bool(filter_overlong)
        self.log_drops = bool(log_drops)
        self.drop_log_every = int(drop_log_every)

    def _render_user_for_gen(self, prompt: str) -> str:
        # Same behavior as trainer._apply_chat_template (user message + generation prompt)
        tok = self.tok
        if tok is not None and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        return prompt

    def _prompt_token_len(self, prompt: str) -> Optional[int]:
        if self.tok is None:
            return None
        text = self._render_user_for_gen(prompt)
        try:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            return len(ids)
        except Exception:
            return None

    def __iter__(self):
        dropped = 0
        kept = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % self.world_size != self.rank:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                msgs = rec.get("messages") or []
                umsgs = [m for m in msgs if m.get("role") == "user"]
                amsgs = [m for m in msgs if m.get("role") == "assistant"]
                if not umsgs:
                    continue
                user = extract_text_from_content(umsgs[0].get("content")).strip()
                ref = extract_text_from_content(amsgs[0].get("content")).strip() if amsgs else ""
                if not user:
                    continue

                # ---- overlong prompt filter ----
                if self.filter_overlong and self.max_prompt_tokens is not None and self.tok is not None:
                    n_tok = self._prompt_token_len(user)
                    if n_tok is not None and n_tok > self.max_prompt_tokens:
                        dropped += 1
                        if self.log_drops and (dropped == 1 or dropped % self.drop_log_every == 0):
                            print(
                                f"[DATA][rank{self.rank}] dropped overlong prompt: "
                                f"{n_tok} > {self.max_prompt_tokens} (dropped={dropped}, kept={kept})",
                                flush=True,
                            )
                        continue

                original_truth = ""
                meta = rec.get("metadata") or {}
                if isinstance(meta, dict):
                    original_truth = (meta.get("original_truth") or "").strip()
                kept += 1
                yield {"prompt": user, "reference": ref, "original_truth": original_truth}


# ============================================================
# Layer class discovery for FSDP wrapping
# ============================================================

def discover_layer_class(config, num_layers: int):
    """Find the Qwen3-Next decoder layer class for FSDP wrapping."""
    wrap_class = None

    # Qwen3-Next class names
    for cls_name in [
        "Qwen3NextDecoderLayer",
        "Qwen3NextMoeDecoderLayer",
    ]:
        try:
            import transformers.models
            for attr_name in dir(transformers.models):
                mod = getattr(transformers.models, attr_name, None)
                if mod and hasattr(mod, cls_name):
                    wrap_class = getattr(mod, cls_name)
                    break
            if wrap_class:
                break
        except Exception:
            continue

    # Broader fallback
    if wrap_class is None:
        for mod_name in ["qwen3_next", "qwen3_next_moe"]:
            try:
                mod = __import__(
                    f"transformers.models.{mod_name}.modeling_{mod_name}",
                    fromlist=["_"],
                )
                for name in dir(mod):
                    if "DecoderLayer" in name:
                        wrap_class = getattr(mod, name)
                        break
                if wrap_class:
                    break
            except ImportError:
                continue

    return wrap_class


def discover_layer_class_from_model(model, num_layers: int):
    """Discover layer class from an instantiated model."""
    for mod in model.modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_layers:
            if len(mod) > 0:
                return type(mod[0])
    return None


# ============================================================
# Judge
# ============================================================

JUDGE_SYSTEM = ""  # 評価スクリプト準拠: system prompt なし

JUDGE_TEMPLATE = (
    "以下は、入力に対するLLMの予測を採点するベンチマークです。\n\n"
    "[入力]\n{prompt}\n\n"
    "[正解（参考）]\n{original_truth}\n\n"
    "[参考: 模範解答]\n{reference}\n\n"
    "[LLMによる予測]\n{candidate}\n\n"
    "採点基準（0-9）:\n"
    "9: 正解とほぼ完全に一致し、重要な要素を網羅\n"
    "7-8: 主要な要素を正しく予測している\n"
    "4-6: 一部は合っているが、重要な要素を見落としている\n"
    "1-3: ほとんど的外れな予測\n"
    "0: 全く無関係または回答なし\n\n"
    "以下のJSON形式で出力すること。json以外は何も出力しないこと。\n"
    "{{\"score\": <0-9の整数>, \"reason\": \"<50文字以内の理由>\"}}"
)


def build_judge_text(prompt: str, candidate: str, reference: Optional[str],
                     original_truth: Optional[str] = None) -> str:
    return JUDGE_TEMPLATE.format(
        prompt=prompt,
        original_truth=original_truth or "(なし)",
        reference=reference or "(なし)",
        candidate=candidate,
    )


class ScoreCache:
    def __init__(self, cache_path: Optional[str]):
        self.path = cache_path
        self.mem = {}
        self._lock = threading.Lock()
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        self.mem[rec["key"]] = rec["score"]
            except Exception:
                pass

    def key(self, prompt: str, candidate: str, reference: str,
            judge_model: str = "") -> str:
        h = hashlib.sha1()
        h.update(prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(candidate.encode("utf-8"))
        h.update(b"\x00")
        h.update((reference or "").encode("utf-8"))
        h.update(b"\x00")
        # Include the judge model so cached scores from a different judge
        # are not reused.
        h.update((judge_model or "").encode("utf-8"))
        return h.hexdigest()

    def get(self, k: str) -> Optional[float]:
        return self.mem.get(k)

    def put(self, k: str, score: float):
        with self._lock:
            self.mem[k] = float(score)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": k, "score": float(score)}, ensure_ascii=False) + "\n")


# ============================================================
# Trainer
# ============================================================

class Qwen3NextFullParamGRPOTrainer:
    """
    Full-parameter GRPO trainer for Qwen3-Next-80B-A3B.

    Unlike the expert-only 397B version, this trains ALL parameters
    (optionally freezing embeddings/lm_head).
    """

    def __init__(self, args):
        self.args = args
        self.wandb_run = None

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required.")

        rank = int(os.environ.get("RANK", "0"))

        # ---- 1) Config & layer class discovery ----
        if rank == 0:
            print("[INIT] Loading model config...", flush=True)

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        num_layers = args.layers
        wrap_class = discover_layer_class(config, num_layers)

        if wrap_class is None:
            with torch.device("meta"):
                try:
                    temp_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                    wrap_class = discover_layer_class_from_model(temp_model, num_layers)
                    del temp_model
                except Exception:
                    pass

        if wrap_class is not None and rank == 0:
            print(f"[INIT] Layer class: {wrap_class.__name__}", flush=True)
        elif rank == 0:
            print("[WARN] Could not find decoder layer class, will discover from loaded model.", flush=True)

        # ---- 2) FSDP / Accelerator ----
        auto_wrap_policy = None
        if wrap_class is not None:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={wrap_class},
            )

        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=True,
            cpu_offload=False,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )

        fsdp_plugin.mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        def _param_init_fn(module):
            has_meta = any(
                p.device.type == "meta"
                for p in module.parameters(recurse=False)
            ) or any(
                b is not None and b.device.type == "meta"
                for b in module.buffers(recurse=False)
            )
            if has_meta:
                module.to_empty(device=torch.device("cpu"), recurse=False)
            return module

        fsdp_plugin.param_init_fn = _param_init_fn

        pg_kwargs = InitProcessGroupKwargs(
            timeout=datetime.timedelta(seconds=7200)
        )

        self.accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=[pg_kwargs],
        )

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device
        self.model_device = self.device

        if self.accelerator.is_main_process:
            print(f"[DIST] rank={self.rank} / world_size={self.world_size}", flush=True)

        self.use_vllm = bool(getattr(args, "use_vllm", False))
        self.vllm_client: Optional[Any] = None
        self.vllm_clients: List = []

        # ---- 3) Tokenizer ----
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # ---- 3.5) prompt token budget for vLLM (input <= max_model_len - gen_max_new_tokens - margin) ----
        self.vllm_max_model_len = int(getattr(args, "vllm_max_model_len", 8192))
        self.prompt_len_margin = int(getattr(args, "prompt_len_margin", 16))
        self.max_prompt_tokens = max(
            1,
            self.vllm_max_model_len - int(self.args.gen_max_new_tokens) - self.prompt_len_margin,
        )
        if self.accelerator.is_main_process:
            print(
                f"[DATA] prompt token budget: <= {self.max_prompt_tokens} "
                f"(vllm_max_model_len={self.vllm_max_model_len}, "
                f"gen_max_new_tokens={int(self.args.gen_max_new_tokens)}, "
                f"margin={self.prompt_len_margin})",
                flush=True,
            )

        # ---- 4) Model loading ----
        if self.rank == 0:
            print(f"[LOAD] rank 0: loading {args.model} BF16 to CPU...", flush=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            print(f"[LOAD] rank {self.rank}: creating meta model...", flush=True)
            import transformers.modeling_utils as _tm_utils

            _orig_load = _tm_utils.PreTrainedModel._load_pretrained_model
            _orig_finalize = _tm_utils.PreTrainedModel._finalize_model_loading
            _orig_check = getattr(_tm_utils, "check_and_set_device_map", None)

            @classmethod
            def _noop_load(cls, model, *a, **kw):
                return {"missing_keys": [], "unexpected_keys": [],
                        "mismatched_keys": [], "error_msgs": []}, None

            @classmethod
            def _noop_finalize(cls, model, *a, **kw):
                return {"missing_keys": [], "unexpected_keys": [],
                        "mismatched_keys": [], "error_msgs": []}

            _tm_utils.PreTrainedModel._load_pretrained_model = _noop_load
            _tm_utils.PreTrainedModel._finalize_model_loading = _noop_finalize
            if _orig_check is not None:
                _tm_utils.check_and_set_device_map = lambda device_map, **kw: device_map

            try:
                with torch.device("meta"):
                    base_model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        config=config,
                        torch_dtype=torch.bfloat16,
                        device_map=None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
            finally:
                _tm_utils.PreTrainedModel._load_pretrained_model = _orig_load
                _tm_utils.PreTrainedModel._finalize_model_loading = _orig_finalize
                if _orig_check is not None:
                    _tm_utils.check_and_set_device_map = _orig_check
            print(f"[LOAD] rank {self.rank}: meta model created.", flush=True)

        # Discover wrap class from model if needed
        if wrap_class is None:
            wrap_class = discover_layer_class_from_model(base_model, num_layers)
            if wrap_class is not None:
                if self.rank == 0:
                    print(f"[FSDP] Discovered wrap class: {wrap_class.__name__}", flush=True)
                fsdp_plugin.auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={wrap_class},
                )

        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = True
        if hasattr(base_model.config, "output_router_logits"):
            base_model.config.output_router_logits = False

        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        base_model.train()

        # ---- 5) Freeze control ----
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        for name, p in base_model.named_parameters():
            total_params += p.numel()
            if args.freeze_embeddings and (
                "embed_tokens" in name or "lm_head" in name
            ):
                p.requires_grad = False
                frozen_params += p.numel()
            else:
                p.requires_grad = True
                trainable_params += p.numel()

        if self.accelerator.is_main_process:
            print(
                f"[INFO] trainable={trainable_params:,} / total={total_params:,} "
                f"({trainable_params / max(1, total_params) * 100:.2f}%) "
                f"frozen={frozen_params:,}",
                flush=True,
            )

        # Log hybrid attention info
        if self.accelerator.is_main_process:
            tc = getattr(config, "text_config", config)
            fai = getattr(tc, "full_attention_interval", None)
            if fai:
                n_full = len([i for i in range(num_layers) if i % fai == 0])
                n_delta = num_layers - n_full
                print(f"[INFO] Hybrid attention: interval={fai} → "
                      f"{n_full} FullAttn + {n_delta} DeltaNet (48 total)", flush=True)

        # ---- 6) Materialize meta buffers ----
        if self.rank != 0:
            for module in base_model.modules():
                for bname in list(module._buffers.keys()):
                    buf = module._buffers[bname]
                    if buf is not None and buf.device.type == "meta":
                        module._buffers[bname] = torch.empty(
                            buf.shape, dtype=buf.dtype, device="cpu")

        self.accelerator.wait_for_everyone()

        # ---- 7) DataLoader ----
        self.ds = ChatMLJsonl(
            args.dataset,
            rank=self.rank,
            world_size=self.world_size,
            tokenizer=self.tok,
            max_prompt_tokens=self.max_prompt_tokens,
            filter_overlong=bool(getattr(args, "filter_overlong_prompts", True)),
            log_drops=bool(getattr(args, "log_overlong_drops", False)) and self.accelerator.is_main_process,
        )
        loader = DataLoader(self.ds, batch_size=args.batch_size, drop_last=True)

        # ---- 8) FSDP wrap ----
        if self.accelerator.is_main_process:
            print("[FSDP] Wrapping model...", flush=True)

        self.model = self.accelerator.prepare(base_model)
        self.loader = loader
        self.is_fsdp = isinstance(self.model, FSDP)

        if self.accelerator.is_main_process:
            print("[FSDP] Model wrapped.", flush=True)

        # ---- 9) Optimizer ----
        train_params = [p for p in self.model.parameters() if p.requires_grad]
        if not train_params:
            raise RuntimeError("No trainable parameters found.")
        self.opt = torch.optim.AdamW(
            train_params,
            lr=args.lr,
            betas=(0.9, 0.98),
            weight_decay=args.weight_decay,
        )

        # Same priority as SFT/DPO: prefer the tokenizer's dedicated pad
        # token; fall back to eos only if pad is missing.
        self.pad_token_id = (
            self.tok.pad_token_id
            if self.tok.pad_token_id is not None
            else self.tok.eos_token_id
        )

        # ---- 10) Judge ----
        if getattr(args, "reward_cache", None):
            base = args.reward_cache
            root, ext = os.path.splitext(base)
            ext = ext if ext else ".jsonl"
            args.reward_cache = f"{root}.rank{self.rank}{ext}"

        self.cache = ScoreCache(args.reward_cache)
        self._thread_local = threading.local()

        # ---- 11) vLLM client ----
        if self.use_vllm:
            if VLLMClient is None:
                raise RuntimeError("VLLMClient not available.")
            self._init_vllm_client()

        self.global_step = 0
        self.baseline: Optional[float] = None

        # ---- 12) W&B ----
        if self.args.wandb and self.accelerator.is_main_process:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.args.wandb_project,
                    entity=self.args.wandb_entity or None,
                    name=self.args.wandb_run_name,
                    config=vars(self.args),
                )
            except Exception as e:
                print(f"[WARN] wandb init failed: {e}", flush=True)
                self.wandb_run = None

        if self.accelerator.is_main_process:
            print("=" * 60)
            print("  Qwen3-Next-80B-A3B Full-Param GRPO")
            print(f"  [v2] Judge: benchmark 0-9 scoring with original_truth")
            print(f"  model           : {self.args.model}")
            print(f"  judge           : {self.args.judge_backend}")
            if self.args.judge_backend != "local":
                print(f"  judge model     : {self.args.judge_model_name}")
                print(f"  judge concurrency: {getattr(self.args, 'judge_concurrency', 1)}")
            print(f"  dataset         : {self.args.dataset}")
            print(f"  layers          : {self.args.layers}")
            print(f"  freeze_embed    : {self.args.freeze_embeddings}")
            print(f"  grpo_group_size : {self.args.grpo_group_size}")
            print(f"  use_vllm        : {self.args.use_vllm}")
            print(f"  vllm_update_every: {getattr(self.args, 'vllm_update_every', 1)}")
            print(f"  wandb           : {self.args.wandb}")
            print("=" * 60, flush=True)

    # ============================================================
    # vLLM Client
    # ============================================================

    def _init_vllm_client(self):
        timeout = self.args.vllm_connection_timeout

        if self.accelerator.is_main_process:
            print("[VLLM] Initializing VLLMClient(s)...", flush=True)

        urls: List[Optional[str]] = []
        if getattr(self.args, "vllm_base_urls", None):
            urls = [u.strip() for u in self.args.vllm_base_urls.split(",") if u.strip()]
        elif self.args.vllm_base_url:
            urls = [self.args.vllm_base_url]
        else:
            urls = [None]

        clients = []
        for u in urls:
            c = VLLMClient(u, connection_timeout=timeout) if u else VLLMClient(connection_timeout=timeout)
            c.init_communicator(self.device)
            clients.append(c)

        # Initial weight sync (full model)
        trainable_only = not self.args.freeze_embeddings  # If nothing frozen, trainable_only=True sends all
        # Actually for full param, trainable_only=True still means all trainable = all params (minus frozen embed)
        for c in clients:
            if self.is_fsdp:
                c.update_model_params_streaming_fsdp(self.model, trainable_only=False)
            else:
                c.update_model_params(self.model, trainable_only=False)
            if self.accelerator.is_main_process:
                print("[VLLM] Initial weight sync OK", flush=True)

        self.vllm_clients = clients
        self.vllm_client = clients[0] if clients else None

    # ============================================================
    # Chat Template
    # ============================================================

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template. Qwen3-Next-Instruct: no thinking mode."""
        if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            try:
                return self.tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        return prompt

    def _apply_template_text(self, messages, add_generation_prompt=True) -> Optional[str]:
        if not (hasattr(self.tok, "apply_chat_template")
                and getattr(self.tok, "chat_template", None)):
            return None
        try:
            return self.tok.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tok.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )

    # ============================================================
    # Generation
    # ============================================================

    @torch.no_grad()
    def _generate_with_vllm_grpo(self, prompts: List[str], group_size: int) -> List[List[str]]:
        clients = self.vllm_clients if self.vllm_clients else [self.vllm_client]
        orig_B = len(prompts)
        force_n1 = bool(getattr(self.args, "vllm_force_n1", False))

        call_prompts = prompts
        call_n = int(group_size)

        if force_n1:
            expanded = []
            for p in prompts:
                expanded.extend([p] * int(group_size))
            call_prompts = expanded
            call_n = 1

        call_prompts = [self._apply_chat_template(p) for p in call_prompts]

        gen_kwargs = {
            "max_tokens": int(self.args.gen_max_new_tokens),
            "repetition_penalty": float(self.args.gen_repetition_penalty),
            "n": int(call_n),
            "temperature": float(self.args.gen_temperature),
            "top_p": float(self.args.gen_top_p),
            "min_p": 0.0,
        }

        def _decode_resp(resp, local_prompts, local_n) -> List[List[str]]:
            if isinstance(resp, dict) and "completion_texts" in resp:
                grouped = resp["completion_texts"]
                result = []
                for i in range(len(local_prompts)):
                    if i < len(grouped):
                        group = grouped[i]
                        if len(group) < int(local_n):
                            group = group + [""] * (int(local_n) - len(group))
                        result.append([t.strip() for t in group[:int(local_n)]])
                    else:
                        result.append([""] * int(local_n))
                return result

            token_seqs = resp.get("completion_ids", resp) if isinstance(resp, dict) else resp
            outs_flat = []

            def _to_list(x):
                return x.tolist() if isinstance(x, torch.Tensor) else x

            def _decode_ids(ids):
                ids = _to_list(ids)
                if not ids:
                    outs_flat.append("")
                    return
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    for sub in ids:
                        try:
                            txt = self.tok.decode(_to_list(sub), skip_special_tokens=True)
                        except Exception:
                            txt = ""
                        outs_flat.append((txt or "").strip())
                    return
                try:
                    txt = self.tok.decode(ids, skip_special_tokens=True)
                except Exception:
                    txt = ""
                outs_flat.append((txt or "").strip())

            def _handle_seq(seq):
                seq = _to_list(seq)
                if seq is None:
                    outs_flat.append("")
                    return
                if isinstance(seq, dict):
                    ids = None
                    for k in ["output_token_ids", "token_ids", "tokens",
                              "output_ids", "generated_token_ids", "completion_ids"]:
                        if k in seq:
                            ids = seq[k]
                            break
                    text = seq.get("text") or seq.get("generated_text")
                    if ids is not None:
                        _decode_ids(ids)
                    else:
                        outs_flat.append((text or "").strip())
                    return
                if isinstance(seq, list) and seq and isinstance(seq[0], (dict, list)):
                    for sub in seq:
                        _handle_seq(sub)
                    return
                _decode_ids(seq)

            _handle_seq(token_seqs)

            B = len(local_prompts)
            expected = B * int(local_n)
            outs_flat_trimmed = outs_flat[:min(len(outs_flat), expected)]

            grouped = []
            idx = 0
            for _ in range(B):
                group = outs_flat_trimmed[idx: idx + int(local_n)]
                if len(group) < int(local_n):
                    group = group + [""] * (int(local_n) - len(group))
                grouped.append(group)
                idx += int(local_n)
            return grouped

        if self.accelerator.is_main_process:
            print(f"[GEN] {len(call_prompts)} prompts × n={call_n}", flush=True)

        _gen_t0 = time.time()

        if len(clients) == 1:
            resp = clients[0].generate(call_prompts, **gen_kwargs)
            grouped = _decode_resp(resp, call_prompts, call_n)
            if self.accelerator.is_main_process:
                print(f"[GEN] done in {time.time() - _gen_t0:.1f}s", flush=True)
            if not force_n1:
                return grouped
            out = []
            ptr = 0
            for _ in range(orig_B):
                g = []
                for _ in range(int(group_size)):
                    g.append((grouped[ptr][0] if grouped[ptr] else "").strip())
                    ptr += 1
                out.append(g)
            return out

        # Multi-server
        K = len(clients)
        buckets = [[] for _ in range(K)]
        idxs = [[] for _ in range(K)]
        for i, p in enumerate(call_prompts):
            k = i % K
            buckets[k].append(p)
            idxs[k].append(i)

        def _run(k):
            if not buckets[k]:
                return k, []
            resp_k = clients[k].generate(buckets[k], **gen_kwargs)
            return k, _decode_resp(resp_k, buckets[k], call_n)

        per_prompt_out = [None] * len(call_prompts)
        with ThreadPoolExecutor(max_workers=K) as ex:
            for k, grouped_k in ex.map(_run, range(K)):
                for pos, g in zip(idxs[k], grouped_k):
                    per_prompt_out[pos] = g

        if self.accelerator.is_main_process:
            print(f"[GEN] done in {time.time() - _gen_t0:.1f}s ({K} servers)", flush=True)

        for i in range(len(per_prompt_out)):
            if per_prompt_out[i] is None:
                per_prompt_out[i] = [""] * int(call_n)

        if not force_n1:
            return per_prompt_out

        out = []
        ptr = 0
        for _ in range(orig_B):
            g = []
            for _ in range(int(group_size)):
                g.append((per_prompt_out[ptr][0] if per_prompt_out[ptr] else "").strip())
                ptr += 1
            out.append(g)
        return out

    @torch.no_grad()
    def generate_candidates_grpo(self, prompts: List[str]) -> List[List[str]]:
        group_size = int(self.args.grpo_group_size)
        if self.vllm_client is not None or self.vllm_clients:
            return self._generate_with_vllm_grpo(prompts, group_size)
        raise RuntimeError("Full-param 80B requires --use_vllm for generation.")

    # ============================================================
    # Judge / Scoring
    # ============================================================

    def _call_external_judge(self, text: str) -> str:
        backend = self.args.judge_backend
        api_base = self.args.judge_api_base
        model_name = self.args.judge_model_name
        api_key = self.args.judge_api_key

        if api_base is None:
            api_base = "https://openrouter.ai/api" if backend == "openrouter" else "http://localhost:8000"

        if api_key is None:
            if backend == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key is None:
                api_key = os.getenv("JUDGE_API_KEY")

        if api_key is None or not model_name:
            return ""

        url = api_base.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if backend == "openrouter":
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = os.environ.get("OPENROUTER_X_TITLE", "grpo-trainer")

        messages = [{"role": "user", "content": text}]
        if JUDGE_SYSTEM:
            messages.insert(0, {"role": "system", "content": JUDGE_SYSTEM})

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": int(self.args.judge_max_new_tokens),
            "temperature": float(self.args.judge_temperature),
        }

        tls = self._thread_local
        if not hasattr(tls, "session"):
            s = requests.Session()
            pool_sz = max(int(getattr(self.args, "judge_concurrency", 100)), 10)
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=pool_sz, pool_maxsize=pool_sz, max_retries=0,
            )
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            tls.session = s
        session = tls.session

        last_error = None
        for attempt in range(1, self.args.judge_max_retries + 1):
            try:
                resp = session.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else min(2.0 ** min(attempt - 1, 5), 30.0)
                    jitter = wait * (0.7 + 0.6 * random.random())
                    time.sleep(jitter)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.exceptions.HTTPError as e:
                last_error = e
                status = getattr(e.response, "status_code", None)
                if status and 500 <= status < 600:
                    time.sleep(min(1.0 * (attempt ** 0.5), 10.0) * (0.5 + random.random()))
                elif status == 429:
                    time.sleep(min(2.0 ** min(attempt - 1, 5), 30.0) * (0.7 + 0.6 * random.random()))
                else:
                    time.sleep(1.0)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                time.sleep(min(2.0 ** min(attempt - 1, 4), 15.0) * (0.5 + random.random()))
            except Exception as e:
                last_error = e
                time.sleep(min(2.0 ** min(attempt - 1, 5), 30.0) * (0.5 + random.random()))

        print(f"[ERROR] judge exhausted {self.args.judge_max_retries} retries: {last_error}", flush=True)
        return ""

    @torch.no_grad()
    def _score_single(self, prompt: str, candidate: str, reference: Optional[str],
                      original_truth: Optional[str] = None) -> Tuple[float, str]:
        text = build_judge_text(prompt, candidate, reference, original_truth=original_truth)
        gen_txt = self._call_external_judge(text)

        if not gen_txt:
            return float("nan"), ""

        # Parse JSON {"score": 0-9, "reason": "..."}
        score_val = 0.0
        try:
            clean = gen_txt.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            parsed = json.loads(clean)
            score_val = float(parsed.get("score", 0))
        except Exception:
            # Fallback: find "score": N pattern
            matches = re.findall(r'"score"\s*:\s*(\d)', gen_txt)
            if matches:
                score_val = float(matches[-1])
            else:
                matches = re.findall(r"\b([0-9])\b", gen_txt)
                if matches:
                    score_val = float(matches[-1])
        score_val = max(0.0, min(9.0, score_val))
        return score_val, gen_txt

    @torch.no_grad()
    def judge_scores_flat(
        self, prompts_flat: List[str], cands_flat: List[str], refs_flat: List[str],
        truths_flat: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        N = len(prompts_flat)
        scores: List[Optional[float]] = [None] * N
        judge_outputs: List[str] = [""] * N

        uncached_indices = []
        for i, (p, c, r) in enumerate(zip(prompts_flat, cands_flat, refs_flat)):
            t = truths_flat[i] if truths_flat else ""
            k = self.cache.key(p, c, r + t, judge_model=self.args.judge_model_name or "")
            cached = self.cache.get(k)
            if cached is not None:
                scores[i] = float(cached)
            else:
                uncached_indices.append(i)

        if self.accelerator.is_main_process and uncached_indices:
            print(f"[JUDGE] {N - len(uncached_indices)} cached, {len(uncached_indices)} to score", flush=True)

        fail_count = 0
        _judge_t0 = time.time()

        if uncached_indices:
            concurrency = int(getattr(self.args, "judge_concurrency", 1))

            def _score_idx(idx: int) -> Tuple[int, float, str]:
                p, c, r = prompts_flat[idx], cands_flat[idx], refs_flat[idx]
                t = truths_flat[idx] if truths_flat else ""
                s, jo = self._score_single(p, c, r, original_truth=t)
                if not math.isnan(s):
                    k = self.cache.key(p, c, r + t, judge_model=self.args.judge_model_name or "")
                    self.cache.put(k, s)
                return idx, s, jo

            workers = min(concurrency, len(uncached_indices))
            with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
                future_to_idx = {ex.submit(_score_idx, i): i for i in uncached_indices}
                done = 0
                for fut in as_completed(future_to_idx):
                    try:
                        idx, s, jo = fut.result()
                        scores[idx] = s
                        judge_outputs[idx] = jo
                        if math.isnan(s):
                            fail_count += 1
                    except Exception as e:
                        idx = future_to_idx[fut]
                        scores[idx] = float("nan")
                        fail_count += 1
                    done += 1
                    if self.accelerator.is_main_process and done == len(uncached_indices):
                        print(f"[JUDGE] scored {done}/{len(uncached_indices)} "
                              f"(fails={fail_count}) in {time.time() - _judge_t0:.1f}s", flush=True)

        for i in range(N):
            if scores[i] is None:
                scores[i] = float("nan")
                fail_count += 1

        if fail_count > 0 and self.accelerator.is_main_process:
            pct = fail_count / max(1, N) * 100
            print(f"[WARN] judge failures: {fail_count}/{N} ({pct:.1f}%)", flush=True)

        return torch.tensor(scores, dtype=torch.float32, device=self.model_device), judge_outputs

    # ============================================================
    # Policy batch
    # ============================================================

    def _build_policy_batch(self, prompts_flat: List[str], cands_flat: List[str]):
        input_ids_list = []
        labels_list = []

        for p, c in zip(prompts_flat, cands_flat):
            text_user = self._apply_template_text(
                [{"role": "user", "content": p}], add_generation_prompt=True,
            )
            if text_user is not None:
                text_full = self._apply_template_text(
                    [{"role": "user", "content": p},
                     {"role": "assistant", "content": c}],
                    add_generation_prompt=False,
                )
                ids_user = self.tok(
                    text_user, return_tensors="pt", add_special_tokens=False,
                )["input_ids"][0]
                ids_full = self.tok(
                    text_full, return_tensors="pt", add_special_tokens=False,
                )["input_ids"][0]
                start = ids_user.shape[0]
                ids = ids_full
                labels = torch.full_like(ids, -100)
                labels[start:] = ids[start:]
            else:
                full_text = p + "\n" + c
                enc = self.tok(full_text, return_tensors="pt")
                ids = enc["input_ids"][0]
                enc_p = self.tok(p, return_tensors="pt", add_special_tokens=False)
                p_len = enc_p["input_ids"].shape[1]
                labels = torch.full_like(ids, -100)
                if p_len < ids.shape[0]:
                    labels[p_len:] = ids[p_len:]
                else:
                    labels[:] = ids

            input_ids_list.append(ids)
            labels_list.append(labels)

        max_len = min(max(x.shape[0] for x in input_ids_list), self.args.seq_len)
        batch_input_ids, batch_labels, batch_attn = [], [], []

        for ids, lbl in zip(input_ids_list, labels_list):
            ids = ids[:max_len]
            lbl = lbl[:max_len]
            pad_len = max_len - ids.shape[0]
            # Build the mask from the real (pre-padding) length rather than
            # `ids != pad_token_id`, which would also mask genuine pad/eos
            # tokens (e.g. <|im_end|>) inside the sequence.
            attn = torch.ones_like(ids)
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=self.pad_token_id)
                lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=-100)
                attn = torch.nn.functional.pad(attn, (0, pad_len), value=0)
            attn = attn.long()
            batch_input_ids.append(ids)
            batch_labels.append(lbl)
            batch_attn.append(attn)

        return (
            torch.stack(batch_input_ids, dim=0).to(self.model_device),
            torch.stack(batch_attn, dim=0).to(self.model_device),
            torch.stack(batch_labels, dim=0).to(self.model_device),
        )

    # ============================================================
    # Distributed helpers
    # ============================================================

    def _gather_strings_to_rank0(self, local_strings: List[str]) -> Optional[List[List[str]]]:
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_strings)
        return gathered if self.rank == 0 else None

    def _broadcast_object_from_rank0(self, obj):
        container = [obj]
        dist.broadcast_object_list(container, src=0)
        return container[0]

    # ============================================================
    # Training step
    # ============================================================

    def step(self, batch):
        prompts: List[str] = batch["prompt"]
        refs: List[str] = batch["reference"]
        truths: List[str] = batch.get("original_truth", [""] * len(batch["prompt"]))
        B = len(prompts)
        if B == 0:
            return
        group_size = int(self.args.grpo_group_size)

        # Phase 1: Gather all prompts to rank 0
        all_prompts_per_rank = self._gather_strings_to_rank0(prompts)
        all_refs_per_rank = self._gather_strings_to_rank0(refs)
        all_truths_per_rank = self._gather_strings_to_rank0(truths)

        # Phase 2: Rank 0 does generation + judging
        if self.rank == 0:
            global_prompts, global_refs, global_truths, rank_sizes = [], [], [], []
            for rk in range(self.world_size):
                rk_prompts = all_prompts_per_rank[rk]
                rk_refs = all_refs_per_rank[rk]
                rk_truths = all_truths_per_rank[rk]
                global_prompts.extend(rk_prompts)
                global_refs.extend(rk_refs)
                global_truths.extend(rk_truths)
                rank_sizes.append(len(rk_prompts))

            broadcast_data = None
            recovery_needed = False

            try:
                all_candidate_groups = self._generate_with_vllm_grpo(global_prompts, group_size)
            except Exception as gen_err:
                if VLLMRecoveryNeeded is not None and isinstance(gen_err, VLLMRecoveryNeeded):
                    recovery_ok = all(
                        c._pynccl_comm is not None for c in self.vllm_clients if c._is_main
                    )
                    if recovery_ok:
                        print(f"[RECOVERY] vLLM recovered, need weight resync: {gen_err}", flush=True)
                        recovery_needed = True
                        all_candidate_groups = None
                    else:
                        print(f"[RECOVERY] vLLM recovery FAILED: {gen_err}", flush=True)
                        broadcast_data = {"skip_step": True}
                else:
                    raise

            if not recovery_needed and not (isinstance(broadcast_data, dict) and broadcast_data.get("skip_step")):
                g_prompts_flat, g_refs_flat, g_cands_flat, g_truths_flat = [], [], [], []
                for p, r, t, group in zip(global_prompts, global_refs, global_truths, all_candidate_groups):
                    for c in group:
                        g_prompts_flat.append(p)
                        g_refs_flat.append(r)
                        g_truths_flat.append(t)
                        g_cands_flat.append(c)

                g_scores_flat, g_judge_outputs_flat = self.judge_scores_flat(
                    g_prompts_flat, g_cands_flat, g_refs_flat, truths_flat=g_truths_flat,
                )

                broadcast_data = {
                    "recovery_needed": False,
                    "rank_sizes": rank_sizes,
                    "all_candidate_groups": all_candidate_groups,
                    "scores_flat": g_scores_flat.cpu().tolist(),
                    "judge_outputs_flat": g_judge_outputs_flat,
                }
            elif not (isinstance(broadcast_data, dict) and broadcast_data.get("skip_step")):
                broadcast_data = {"recovery_needed": True}
        else:
            broadcast_data = None

        # Phase 3: Broadcast
        broadcast_data = self._broadcast_object_from_rank0(broadcast_data)

        if broadcast_data.get("skip_step", False):
            return {}

        # Handle recovery
        if broadcast_data.get("recovery_needed", False):
            if self.accelerator.is_main_process:
                print("[RECOVERY] All ranks: weight resync...", flush=True)
            for ci, c in enumerate(self.vllm_clients):
                if self.is_fsdp:
                    c.update_model_params_streaming_fsdp(self.model, trainable_only=False)
                else:
                    c.update_model_params(self.model, trainable_only=False)

            if self.rank == 0:
                try:
                    all_candidate_groups = self._generate_with_vllm_grpo(global_prompts, group_size)
                    g_prompts_flat, g_refs_flat, g_cands_flat, g_truths_flat = [], [], [], []
                    for p, r, t, group in zip(global_prompts, global_refs, global_truths, all_candidate_groups):
                        for c in group:
                            g_prompts_flat.append(p)
                            g_refs_flat.append(r)
                            g_truths_flat.append(t)
                            g_cands_flat.append(c)
                    g_scores_flat, g_judge_outputs_flat = self.judge_scores_flat(
                        g_prompts_flat, g_cands_flat, g_refs_flat, truths_flat=g_truths_flat,
                    )
                    broadcast_data = {
                        "recovery_needed": False,
                        "rank_sizes": rank_sizes,
                        "all_candidate_groups": all_candidate_groups,
                        "scores_flat": g_scores_flat.cpu().tolist(),
                        "judge_outputs_flat": g_judge_outputs_flat,
                    }
                except Exception as retry_err:
                    print(f"[RECOVERY] Retry failed: {retry_err}", flush=True)
                    broadcast_data = {"skip_step": True}
            else:
                broadcast_data = None

            broadcast_data = self._broadcast_object_from_rank0(broadcast_data)
            if broadcast_data.get("skip_step", False):
                return {}

        rank_sizes = broadcast_data["rank_sizes"]
        all_candidate_groups = broadcast_data["all_candidate_groups"]
        all_scores_flat = broadcast_data["scores_flat"]
        all_judge_outputs_flat = broadcast_data["judge_outputs_flat"]

        offset = sum(rank_sizes[:self.rank])
        my_B = rank_sizes[self.rank]
        candidate_groups = all_candidate_groups[offset:offset + my_B]

        flat_offset = offset * group_size
        flat_count = my_B * group_size
        scores_flat = torch.tensor(
            all_scores_flat[flat_offset:flat_offset + flat_count],
            dtype=torch.float32, device=self.model_device,
        )
        judge_outputs_flat = all_judge_outputs_flat[flat_offset:flat_offset + flat_count]

        prompts_flat, refs_flat, cands_flat = [], [], []
        for p, r, group in zip(prompts, refs, candidate_groups):
            for c in group:
                prompts_flat.append(p)
                refs_flat.append(r)
                cands_flat.append(c)

        # Phase 4: GRPO
        nan_mask = torch.isnan(scores_flat)
        n_nan = nan_mask.sum().item()
        scores_clean = scores_flat.clone()
        if n_nan > 0:
            scores_clean[nan_mask] = 0.0

        mean_score = scores_clean[~nan_mask].mean().item() if (~nan_mask).any() else 0.0
        if self.baseline is None:
            self.baseline = mean_score
        else:
            self.baseline = (1 - self.args.ema_alpha) * self.baseline + self.args.ema_alpha * mean_score

        scores_group = scores_clean.view(B, group_size)
        nan_group = nan_mask.view(B, group_size)
        valid_group = ~nan_group

        valid_count = valid_group.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_group = (scores_group * valid_group.float()).sum(dim=1, keepdim=True) / valid_count
        diff = (scores_group - mean_group) * valid_group.float()
        var_group = (diff ** 2).sum(dim=1, keepdim=True) / valid_count.clamp_min(2.0)
        std_group = var_group.sqrt()

        adv_group = (scores_group - mean_group) / (std_group + 1e-6)
        adv_group = adv_group * valid_group.float()
        too_few = (valid_count < 2).expand_as(adv_group)
        adv_group[too_few] = 0.0

        if self.args.reward_scale is not None:
            adv_group = adv_group * self.args.reward_scale
        if self.args.reward_clip > 0:
            adv_group = adv_group.clamp(-self.args.reward_clip, self.args.reward_clip)
        advantage = adv_group.view(-1)

        input_ids, attention_mask, labels = self._build_policy_batch(prompts_flat, cands_flat)
        _train_t0 = time.time()
        self.model.train()
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_router_logits=False,
            return_dict=True,
        )
        logits = out.logits[:, :-1, :]
        labels_shifted = labels[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        valid_mask = (labels_shifted != -100).float()
        safe_labels = labels_shifted.clone()
        safe_labels[safe_labels == -100] = 0
        token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid_mask
        lengths = valid_mask.sum(dim=-1).clamp_min(1.0)
        logp_mean = token_logp.sum(dim=-1) / lengths

        valid_flat = (~nan_mask).float()
        n_valid = valid_flat.sum().clamp_min(1.0)
        loss = -(advantage * logp_mean * valid_flat).sum() / n_valid

        self.opt.zero_grad(set_to_none=True)
        self.accelerator.backward(loss)
        if self.args.grad_clip > 0:
            self.accelerator.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.args.grad_clip,
            )
        self.opt.step()
        _train_elapsed = time.time() - _train_t0

        current_step = self.global_step + 1

        # Sync to vLLM
        _sync_t0 = time.time()
        if self.vllm_clients:
            if current_step % int(getattr(self.args, "vllm_update_every", 1)) == 0:
                for ci, c in enumerate(self.vllm_clients):
                    if self.is_fsdp:
                        c.update_model_params_streaming_fsdp(self.model, trainable_only=False)
                    else:
                        c.update_model_params(self.model, trainable_only=False)
                    if self.accelerator.is_main_process:
                        print(f"[VLLM] weight sync OK (step={current_step}, client={ci}) "
                              f"in {time.time() - _sync_t0:.1f}s", flush=True)

        # Logging
        if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
            valid_scores = scores_clean[~nan_mask] if n_nan > 0 else scores_flat
            r_mean = valid_scores.mean().item() if valid_scores.numel() > 0 else 0.0
            r_std = valid_scores.std().item() if valid_scores.numel() > 1 else 0.0
            fail_str = f" fails={int(n_nan)}" if n_nan > 0 else ""
            total_steps = int(self.args.steps)
            elapsed = time.time() - self._train_start_time
            avg_per_step = elapsed / max(current_step, 1)
            remaining = avg_per_step * (total_steps - current_step)
            eta_m, eta_s = divmod(int(remaining), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h > 0 else f"{eta_m}m{eta_s:02d}s"
            print(
                f"[step {current_step}/{total_steps}] loss={loss.item():.4f} "
                f"score={r_mean:.3f}±{r_std:.3f} "
                f"baseline={self.baseline:.3f}{fail_str} "
                f"train={_train_elapsed:.1f}s ETA={eta_str}",
                flush=True,
            )

        if (self.wandb_run is not None
                and self.accelerator.is_main_process
                and current_step % self.args.log_every == 0):
            try:
                import wandb
                valid_scores = scores_clean[~nan_mask] if n_nan > 0 else scores_flat
                r_mean = float(valid_scores.mean().item()) if valid_scores.numel() > 0 else 0.0
                r_std = float(valid_scores.std().item()) if valid_scores.numel() > 1 else 0.0
                group_std = std_group.squeeze(1)
                frac_reward_zero_std = float((group_std < 1e-6).float().mean().item())

                log_dict = {
                    "loss": float(loss.item()),
                    "reward_mean": r_mean,
                    "reward_std": r_std,
                    "baseline": float(self.baseline),
                    "advantage_mean": float(advantage.mean().item()),
                    "logp_mean": float(logp_mean.mean().item()),
                    "frac_reward_zero_std": frac_reward_zero_std,
                    "response_length/mean": float(lengths.mean().item()),
                    "judge/fail_count": int(n_nan),
                    "train/global_step": current_step,
                    "train/batch_size_global": B * self.world_size,
                    "timing/train_s": _train_elapsed,
                }
                self.wandb_run.log(log_dict, step=current_step)

                table = wandb.Table(
                    columns=["step", "idx", "group_idx", "prompt", "reference",
                             "candidate", "judge_output", "score", "advantage"]
                )
                for i in range(B):
                    for g_idx in range(group_size):
                        idx_flat = i * group_size + g_idx
                        table.add_data(
                            current_step, i, g_idx, prompts[i], refs[i],
                            cands_flat[idx_flat], judge_outputs_flat[idx_flat],
                            float(scores_flat[idx_flat].item()),
                            float(advantage[idx_flat].item()),
                        )
                self.wandb_run.log({"samples": table}, step=current_step)
            except Exception as e:
                print(f"[WARN] wandb logging failed: {e}", flush=True)

        self.global_step += 1

    # ============================================================
    # Save
    # ============================================================

    @torch.no_grad()
    def save_model_sharded(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        if self.is_fsdp:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                sd = self.model.state_dict()

            path = os.path.join(save_dir, f"model_sharded.rank{self.rank}.pt")
            torch.save(sd, path)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {
                    "format": "full_model_sharded",
                    "pattern": "model_sharded.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model": self.args.model,
                    "arch": "Qwen3-Next-80B-A3B",
                    "layers": self.args.layers,
                    "full_param": True,
                    "steps": int(self.global_step),
                }
                with open(os.path.join(save_dir, "model_sharded_meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE] sharded -> {save_dir}", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped = self.accelerator.unwrap_model(self.model)
                sd = unwrapped.state_dict()
                path = os.path.join(save_dir, "model.pt")
                torch.save(sd, path)
                print(f"[SAVE] full model -> {path}", flush=True)

    # ============================================================
    # Train loop
    # ============================================================

    def train(self):
        self._train_start_time = time.time()

        if int(getattr(self.args, "epochs", 0)) > 0:
            epochs = int(self.args.epochs)
            global_batch = int(self.world_size) * int(self.args.batch_size)
            try:
                with open(self.args.dataset, "r", encoding="utf-8") as f:
                    dataset_size = sum(1 for _ in f)
            except Exception:
                dataset_size = 0
            if dataset_size <= 0:
                raise RuntimeError(f"Cannot count dataset: {self.args.dataset}")
            steps_per_epoch = (dataset_size + global_batch - 1) // global_batch
            self.args.steps = epochs * steps_per_epoch

            step = 0
            it = iter(self.loader)
            for _ep in range(epochs):
                for _ in range(steps_per_epoch):
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(self.loader)
                        batch = next(it)
                    self.step(batch)
                    step += 1
        else:
            step = 0
            while step < self.args.steps:
                progressed = False
                for batch in self.loader:
                    progressed = True
                    self.step(batch)
                    step += 1
                    if step >= self.args.steps:
                        break
                if not progressed:
                    raise RuntimeError(
                        "DataLoader yielded no batches. "
                        "All samples may have been filtered out as overlong. "
                        "Consider --no_filter_overlong_prompts or adjust "
                        "--vllm_max_model_len/--gen_max_new_tokens/--prompt_len_margin."
                    )

        # Cleanup
        if self.vllm_clients:
            for c in self.vllm_clients:
                try:
                    c.close_communicator()
                except Exception:
                    pass

        self.save_model_sharded(self.args.save_dir)
        self.accelerator.wait_for_everyone()


def build_argparser():
    ap = argparse.ArgumentParser(description="Qwen3-Next-80B Full-Param GRPO (FSDP)")

    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--layers", type=int, default=48, help="Number of layers (Qwen3-Next=48)")

    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=4096)

    ap.add_argument("--freeze_embeddings", action="store_true",
                    help="Freeze embed_tokens and lm_head")

    # GRPO
    ap.add_argument("--grpo_group_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Generation
    ap.add_argument("--gen_max_new_tokens", type=int, default=1024)
    ap.add_argument("--gen_temperature", type=float, default=0.7)
    ap.add_argument("--gen_top_p", type=float, default=0.8)
    ap.add_argument("--gen_repetition_penalty", type=float, default=1.05)

    # vLLM
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_base_url", type=str, default=None)
    ap.add_argument("--vllm_base_urls", type=str, default=None)
    ap.add_argument("--vllm_update_every", type=int, default=1)
    ap.add_argument("--vllm_force_n1", action="store_true")
    ap.add_argument("--vllm_connection_timeout", type=int, default=600)
    ap.add_argument("--vllm_max_model_len", type=int, default=8192,
                    help="Must match vLLM server --max-model-len")
    ap.add_argument("--prompt_len_margin", type=int, default=16,
                    help="Safety margin tokens for template/special tokens etc.")

    # Overlong prompt filter (ON by default)
    ap.set_defaults(filter_overlong_prompts=True)
    ap.add_argument("--no_filter_overlong_prompts", dest="filter_overlong_prompts",
                    action="store_false", help="Do not drop overlong prompts")
    ap.add_argument("--log_overlong_drops", action="store_true",
                    help="Print drop logs for overlong prompts (rank0 only)")

    # Judge
    ap.add_argument("--judge_max_new_tokens", type=int, default=2048)
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_backend", type=str, default="openrouter",
                    choices=["openrouter", "openai"])
    ap.add_argument("--judge_api_base", type=str, default=None)
    ap.add_argument("--judge_api_key", type=str, default=None)
    ap.add_argument("--judge_model_name", type=str, default=None)
    ap.add_argument("--judge_max_retries", type=int, default=1000)
    ap.add_argument("--judge_concurrency", type=int, default=100)

    # Reward
    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--reward_clip", type=float, default=1.0)
    ap.add_argument("--ema_alpha", type=float, default=0.05)
    ap.add_argument("--reward_cache", type=str, default="judge_score_cache.jsonl")

    # Logging
    ap.add_argument("--debug_print", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="qwen3next_fullparam_grpo")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=1)

    # Save
    ap.add_argument("--save_dir", type=str, default="ckpts_qwen3next_grpo")

    return ap


def main():
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    set_seed(42)
    trainer = Qwen3NextFullParamGRPOTrainer(args)
    trainer.train()
    if trainer.wandb_run is not None and trainer.accelerator.is_main_process:
        try:
            trainer.wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
