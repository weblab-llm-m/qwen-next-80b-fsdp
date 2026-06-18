#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-parameter DPO for Qwen3-Next-80B-A3B (FSDP, BF16).

Adapted from Qwen3.5-397B partial-layer DPO.

Changes from 397B partial-layer:
  - All parameters trainable (optional --freeze_embeddings)
  - No LoRA / partial-layer / detach hooks
  - Layer class: Qwen3NextDecoderLayer (48 layers)
  - Gradient checkpointing enabled (full-param memory)
  - Save: full model sharded state dict

All DPO fixes preserved:
  FIX 1: Collator pads chosen+rejected to SAME max_len (torch.cat compat)
  FIX 2: Ref precompute uses concatenated forward (MoE routing consistency)
  FIX 3: ref_bs forced to 1 (batch=2 matches training exactly)
  FIX 4: step() uses model.eval() (dropout consistency with ref)
  FIX 5: clip_grad_norm_ guarded by sync_gradients (accumulation safety)
  FIX 6: GradientAccumulationPlugin with sync_each_batch=True (FSDP OOM)
  FIX 7: Checkpointed cross-entropy (avoids (B*T,V) float32 softmax)

Model: Qwen3-Next-80B-A3B-Instruct
  - 48 layers, hybrid attention (full_attention_interval=4)
  - 512 MoE experts (10 routed + 1 shared)
  - BF16 native, ~80B total / 3B active

Memory (5 nodes = 40 GPU, FSDP FULL_SHARD):
  Weights+Grads+Adam : ~24 GB/GPU
  DPO concat fwd     : 2x batch → ~4-6 GB activations (grad ckpt)
  Total              : ~30 GB/GPU (H100 80GB → comfortable)
"""

import os
import json
import math
import random
import argparse
import datetime
import time
from typing import List, Dict, Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin  # FIX 6
from torch.distributed.fsdp import (
    ShardingStrategy,
    StateDictType,
    ShardedStateDictConfig,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _FSDP_AVAILABLE = True
except Exception:
    _FSDP_AVAILABLE = False


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_sdp_backends():
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


# ============================================================
# Layer class discovery
# ============================================================

def discover_layer_class(config, num_layers: int):
    wrap_class = None
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
    for mod in model.modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_layers:
            if len(mod) > 0:
                return type(mod[0])
    return None


# ============================================================
# DPO Dataset
# ============================================================

class DPOMapDataset(Dataset):
    """Map-style dataset for DPO training."""

    def __init__(self, path: str, tok, seq_len: int):
        self.tok = tok
        self.seq_len = seq_len
        self.samples: List[Dict] = []

        self.ref_chosen_logps: Optional[List[float]] = None
        self.ref_rejected_logps: Optional[List[float]] = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = self._tokenize_pair(obj)
                if result is not None:
                    self.samples.append(result)

        print(f"[DPO-DS] Loaded {len(self.samples)} pairs from {path}", flush=True)

    def _build_messages(self, prompt: str, response: str):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    def _tokenize_with_labels(self, messages) -> Optional[Dict]:
        """Tokenise message list; mask non-assistant tokens in labels."""
        try:
            formatted = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            try:
                formatted = self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                return None
        except Exception:
            return None

        if not formatted:
            return None

        enc = self.tok(formatted, add_special_tokens=False)
        input_ids = enc["input_ids"]
        if not input_ids:
            return None

        labels = [-100] * len(input_ids)

        assistant_header = "<|im_start|>assistant\n"
        im_end_str = "<|im_end|>"
        pos = 0
        while True:
            start = formatted.find(assistant_header, pos)
            if start == -1:
                break
            content_start = start + len(assistant_header)
            end = formatted.find(im_end_str, content_start)
            if end == -1:
                end = len(formatted)
            content_end = end + len(im_end_str)

            prefix_before = formatted[:content_start]
            prefix_after = formatted[:content_end]
            tok_start = len(self.tok(prefix_before, add_special_tokens=False)["input_ids"])
            tok_end = len(self.tok(prefix_after, add_special_tokens=False)["input_ids"])

            for j in range(tok_start, min(tok_end, len(labels))):
                labels[j] = input_ids[j]
            pos = content_end

        if all(l == -100 for l in labels):
            labels = list(input_ids)

        input_ids = input_ids[:self.seq_len]
        labels = labels[:self.seq_len]

        return {"input_ids": input_ids, "labels": labels}

    def _tokenize_pair(self, obj: dict) -> Optional[Dict]:
        prompt = obj.get("prompt")
        chosen = obj.get("chosen")
        rejected = obj.get("rejected")

        if not prompt or not chosen or not rejected:
            return None

        chosen_msgs = self._build_messages(prompt, chosen)
        rejected_msgs = self._build_messages(prompt, rejected)

        chosen_tok = self._tokenize_with_labels(chosen_msgs)
        rejected_tok = self._tokenize_with_labels(rejected_msgs)

        if chosen_tok is None or rejected_tok is None:
            return None

        n_chosen_train = sum(1 for l in chosen_tok["labels"] if l != -100)
        n_rejected_train = sum(1 for l in rejected_tok["labels"] if l != -100)
        if n_chosen_train == 0 or n_rejected_train == 0:
            return None

        return {
            "chosen_input_ids": chosen_tok["input_ids"],
            "chosen_labels": chosen_tok["labels"],
            "rejected_input_ids": rejected_tok["input_ids"],
            "rejected_labels": rejected_tok["labels"],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        result = {
            "chosen_input_ids": s["chosen_input_ids"],
            "chosen_labels": s["chosen_labels"],
            "rejected_input_ids": s["rejected_input_ids"],
            "rejected_labels": s["rejected_labels"],
            "index": idx,
        }
        if self.ref_chosen_logps is not None:
            result["ref_chosen_logps"] = self.ref_chosen_logps[idx]
            result["ref_rejected_logps"] = self.ref_rejected_logps[idx]
        return result


# ============================================================
# FIX 1: Collator pads chosen+rejected to the SAME max_len
# ============================================================

def dpo_collate_fn(batch, pad_id, seq_len):
    all_c_ids = [b["chosen_input_ids"] for b in batch]
    all_c_labels = [b["chosen_labels"] for b in batch]
    all_r_ids = [b["rejected_input_ids"] for b in batch]
    all_r_labels = [b["rejected_labels"] for b in batch]

    c_max = max(len(ids) for ids in all_c_ids)
    r_max = max(len(ids) for ids in all_r_ids)
    max_len = min(max(c_max, r_max), seq_len)

    def _pad_to(all_ids, all_labels):
        ids_t = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask_t = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels_t = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (ids, labs) in enumerate(zip(all_ids, all_labels)):
            slen = min(len(ids), max_len)
            ids_t[i, :slen] = torch.tensor(ids[:slen], dtype=torch.long)
            mask_t[i, :slen] = 1
            labels_t[i, :slen] = torch.tensor(labs[:slen], dtype=torch.long)
        return ids_t, mask_t, labels_t

    c_ids, c_mask, c_labels = _pad_to(all_c_ids, all_c_labels)
    r_ids, r_mask, r_labels = _pad_to(all_r_ids, all_r_labels)

    result = {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }

    if "ref_chosen_logps" in batch[0]:
        result["ref_chosen_logps"] = torch.tensor(
            [b["ref_chosen_logps"] for b in batch], dtype=torch.float32)
        result["ref_rejected_logps"] = torch.tensor(
            [b["ref_rejected_logps"] for b in batch], dtype=torch.float32)

    return result


# ============================================================
# FIX 7: Checkpointed cross-entropy (memory-efficient log-probs)
# ============================================================

def _cross_entropy_no_save(logits_flat: torch.Tensor, labels_flat: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits_flat, labels_flat, reduction='none')


def compute_per_sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = (shift_labels != -100)
    B, T, V = shift_logits.shape

    per_token_ce = torch.utils.checkpoint.checkpoint(
        _cross_entropy_no_save,
        shift_logits.view(-1, V),
        shift_labels.clamp(min=0).view(-1),
        use_reentrant=False,
    ).view(B, T)

    per_token_logps = -per_token_ce * mask.float()
    return per_token_logps.sum(dim=-1)


def compute_per_sequence_logps_with_length(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = (shift_labels != -100)
    B, T, V = shift_logits.shape

    per_token_ce = torch.utils.checkpoint.checkpoint(
        _cross_entropy_no_save,
        shift_logits.view(-1, V),
        shift_labels.clamp(min=0).view(-1),
        use_reentrant=False,
    ).view(B, T)

    per_token_logps = -per_token_ce * mask.float()
    seq_logps = per_token_logps.sum(dim=-1)
    seq_lengths = mask.float().sum(dim=-1)
    return seq_logps, seq_lengths


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    length_normalise: bool = False,
    chosen_lengths: Optional[torch.Tensor] = None,
    rejected_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pc = policy_chosen_logps
    pr = policy_rejected_logps
    rc = ref_chosen_logps.to(pc.device)
    rr = ref_rejected_logps.to(pr.device)

    if length_normalise and chosen_lengths is not None and rejected_lengths is not None:
        pc = pc / chosen_lengths.clamp(min=1).to(pc.device)
        pr = pr / rejected_lengths.clamp(min=1).to(pr.device)
        rc = rc / chosen_lengths.clamp(min=1).to(rc.device)
        rr = rr / rejected_lengths.clamp(min=1).to(rr.device)

    chosen_rewards = beta * (pc - rc)
    rejected_rewards = beta * (pr - rr)

    if loss_type == "sigmoid":
        logits = chosen_rewards - rejected_rewards
        if label_smoothing > 0:
            # TRL convention: -(1-ls)*logsigmoid(logits) - ls*logsigmoid(-logits)
            loss = (
                (1 - label_smoothing) * F.logsigmoid(logits)
                + label_smoothing * F.logsigmoid(-logits)
            )
            loss = -loss.mean()
        else:
            loss = -F.logsigmoid(logits).mean()
    elif loss_type == "hinge":
        loss = torch.relu(1.0 - (chosen_rewards - rejected_rewards)).mean()
    elif loss_type == "ipo":
        logits = (pc - pr) - (rc - rr)
        loss = ((logits - 1.0 / (2.0 * beta)) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, chosen_rewards.detach(), rejected_rewards.detach()


# ============================================================
# DPO Trainer (full-parameter)
# ============================================================

class DPOTrainer:
    def __init__(self, args):
        set_seed(args.seed)
        configure_sdp_backends()
        self.args = args
        self.global_step = 0
        self.global_epoch = 0
        self.use_wandb = False

        # 1) Config & layer class
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        wrap_class = discover_layer_class(config, args.num_layers)

        rank = int(os.environ.get("RANK", "0"))
        if wrap_class is not None and rank == 0:
            print(f"[FSDP] Auto-wrap class: {wrap_class.__name__}", flush=True)
        elif rank == 0:
            print("[WARN] Could not find decoder layer class.", flush=True)

        # 2) FSDP / Accelerator
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
            use_orig_params=True,
            activation_checkpointing=True,
            limit_all_gathers=True,
            cpu_offload=False,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )

        if wrap_class is not None:
            fsdp_plugin.auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={wrap_class},
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

        pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))

        # FIX 6: sync_each_batch=True prevents FSDP grad buffer OOM
        ga_plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=True,
        )

        self.accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_plugin=ga_plugin,
            kwargs_handlers=[pg_kwargs],
        )

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            print(f"[DIST] rank={self.rank} / world_size={self.world_size}", flush=True)

        # 3) Tokenizer & Model
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        if self.rank == 0:
            print(f"[LOAD] rank 0: loading {args.model} BF16 to CPU...", flush=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model, config=config, device_map=None,
                trust_remote_code=True, low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
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
                        args.model, config=config, device_map=None,
                        trust_remote_code=True, low_cpu_mem_usage=True,
                        torch_dtype=torch.bfloat16,
                    )
            finally:
                _tm_utils.PreTrainedModel._load_pretrained_model = _orig_load
                _tm_utils.PreTrainedModel._finalize_model_loading = _orig_finalize
                if _orig_check is not None:
                    _tm_utils.check_and_set_device_map = _orig_check
            print(f"[LOAD] rank {self.rank}: meta model created.", flush=True)

        # Discover from model if needed
        if wrap_class is None:
            wrap_class = discover_layer_class_from_model(base_model, args.num_layers)
            if wrap_class is not None:
                if self.rank == 0:
                    print(f"[FSDP] Discovered: {wrap_class.__name__}", flush=True)
                fsdp_plugin.auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={wrap_class},
                )

        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
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

        # 4) Freeze control — full-param, optional embed freeze
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
            tc = getattr(config, "text_config", config)
            fai = getattr(tc, "full_attention_interval", None)
            if fai:
                n_full = len([i for i in range(args.num_layers) if i % fai == 0])
                n_delta = args.num_layers - n_full
                print(f"[INFO] Hybrid attention: interval={fai} -> "
                      f"{n_full} FullAttn + {n_delta} DeltaNet ({args.num_layers} total)",
                      flush=True)

        # Materialize meta buffers on non-rank-0
        if self.rank != 0:
            for module in base_model.modules():
                for bname in list(module._buffers.keys()):
                    buf = module._buffers[bname]
                    if buf is not None and buf.device.type == "meta":
                        module._buffers[bname] = torch.empty(
                            buf.shape, dtype=buf.dtype, device="cpu")

        self.accelerator.wait_for_everyone()

        # 5) FSDP wrap
        if self.accelerator.is_main_process:
            print("[FSDP] Wrapping model...", flush=True)

        self.model = self.accelerator.prepare(base_model)
        self.is_fsdp = isinstance(self.model, FSDP)

        if self.accelerator.is_main_process:
            print("[FSDP] Model wrapped.", flush=True)

        # 6) Optimizer
        train_params = [p for p in self.model.parameters() if p.requires_grad]
        if not train_params:
            raise RuntimeError("No trainable parameters found.")

        self.opt = torch.optim.AdamW(
            train_params, lr=args.lr, betas=(0.9, 0.98),
            weight_decay=args.weight_decay, foreach=False,
        )
        self.opt = self.accelerator.prepare(self.opt)

        # Pad token
        if self.tok.pad_token_id is not None:
            self.pad_token_id = int(self.tok.pad_token_id)
        elif self.tok.eos_token_id is not None:
            self.pad_token_id = int(self.tok.eos_token_id)
        else:
            raise RuntimeError("Tokenizer has no pad/eos token id.")

        # 7) Dataset
        if self.accelerator.is_main_process:
            print(f"[DATA] dataset: {args.dataset}", flush=True)

        self.dataset = DPOMapDataset(args.dataset, tok=self.tok, seq_len=args.seq_len)

        if self.accelerator.is_main_process and len(self.dataset) > 0:
            s0 = self.dataset.samples[0]
            for side in ("chosen", "rejected"):
                ids = s0[f"{side}_input_ids"]
                labs = s0[f"{side}_labels"]
                n_train = sum(1 for l in labs if l != -100)
                n_mask = sum(1 for l in labs if l == -100)
                print(f"[DATA] sample 0 {side}: {len(ids)} tokens, "
                      f"{n_train} trainable, {n_mask} masked", flush=True)

        # 8) Precompute reference log-probs
        if self.accelerator.is_main_process:
            print(f"\n[REF] Precomputing reference log-probs "
                  f"({len(self.dataset)} pairs)...", flush=True)

        self._precompute_ref_logprobs()

        if self.accelerator.is_main_process:
            rc = self.dataset.ref_chosen_logps
            rr = self.dataset.ref_rejected_logps
            avg_c = sum(rc) / max(len(rc), 1)
            avg_r = sum(rr) / max(len(rr), 1)
            print(f"[REF] Done. avg ref_chosen_logp={avg_c:.2f}, "
                  f"avg ref_rejected_logp={avg_r:.2f}", flush=True)

        # 9) Training DataLoader
        self.train_sampler = DistributedSampler(
            self.dataset, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        _collate = partial(dpo_collate_fn, pad_id=self.pad_token_id, seq_len=args.seq_len)
        self.loader = DataLoader(
            self.dataset, batch_size=args.batch_size,
            sampler=self.train_sampler, collate_fn=_collate,
            num_workers=2, pin_memory=True, drop_last=True,
        )

        gas = args.gradient_accumulation_steps
        eff_batch = args.batch_size * gas * self.world_size
        samples_per_rank = len(self.dataset) // self.world_size
        micro_per_epoch = max(1, samples_per_rank // args.batch_size)
        optim_per_epoch = micro_per_epoch // gas
        total_steps = optim_per_epoch * (args.epochs or 1)
        self.estimated_total_steps = max(1, total_steps)

        if self.accelerator.is_main_process:
            print(f"[DATA] total_pairs={len(self.dataset)}, "
                  f"micro_batches/epoch~={micro_per_epoch}, "
                  f"optim_steps/epoch~={optim_per_epoch}, "
                  f"estimated_total_steps~={total_steps}, "
                  f"effective_batch_size={eff_batch}", flush=True)

        # Summary
        if self.accelerator.is_main_process:
            print("=" * 56)
            print("  Qwen3-Next-80B-A3B Full-Param DPO")
            print(f"  model          : {args.model}")
            print(f"  num_layers     : {args.num_layers}")
            print(f"  freeze_embed   : {args.freeze_embeddings}")
            print(f"  dataset        : {args.dataset}")
            print(f"  batch_size     : {args.batch_size} (micro, per rank)")
            print(f"  grad_accum     : {gas}")
            print(f"  eff_batch      : {eff_batch}")
            print(f"  seq_len        : {args.seq_len}")
            print(f"  beta           : {args.beta}")
            print(f"  loss_type      : {args.loss_type}")
            print(f"  label_smooth   : {args.label_smoothing}")
            print(f"  len_normalise  : {args.length_normalise}")
            print(f"  lr             : {args.lr}")
            print(f"  epochs         : {args.epochs}")
            print(f"  steps(cap)     : {args.steps}")
            print(f"  save_dir       : {args.save_dir}")
            print("=" * 56, flush=True)

        # W&B
        if self.accelerator.is_main_process and args.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or None,
                    config=vars(args),
                    resume="allow",
                )
                self.use_wandb = True
                print(f"[WANDB] Initialized: project={args.wandb_project}", flush=True)
            except Exception as e:
                print(f"[WANDB] Init failed: {e}", flush=True)

    # ==========================================================
    # FIX 2 & 3: Concatenated forward + ref_bs=1
    # ==========================================================

    @torch.no_grad()
    def _precompute_ref_logprobs(self):
        self.model.eval()

        n = len(self.dataset)
        n_padded = math.ceil(n / self.world_size) * self.world_size
        all_indices = list(range(n)) + [0] * (n_padded - n)

        per_rank = n_padded // self.world_size
        my_start = self.rank * per_rank
        my_end = my_start + per_rank
        my_indices = all_indices[my_start:my_end]

        # FIX 3: Force ref_bs=1
        ref_bs = 1
        _collate = partial(dpo_collate_fn, pad_id=self.pad_token_id, seq_len=self.args.seq_len)

        local_results = {}
        t0 = time.time()
        num_batches = math.ceil(len(my_indices) / ref_bs)

        for bi in range(num_batches):
            batch_indices = my_indices[bi * ref_bs : (bi + 1) * ref_bs]
            batch_items = [self.dataset[i] for i in batch_indices]
            batch = _collate(batch_items)

            # FIX 2: Concatenated forward
            c_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
            c_mask = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
            c_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
            r_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
            r_mask = batch["rejected_attention_mask"].to(self.device, non_blocking=True)
            r_labels = batch["rejected_labels"].to(self.device, non_blocking=True)

            all_ids = torch.cat([c_ids, r_ids], dim=0)
            all_mask = torch.cat([c_mask, r_mask], dim=0)
            all_labels = torch.cat([c_labels, r_labels], dim=0)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=all_ids, attention_mask=all_mask,
                    use_cache=False, output_router_logits=False, return_dict=True,
                )
            all_logps = compute_per_sequence_logps(outputs.logits, all_labels)
            bs = c_ids.size(0)
            c_logps = all_logps[:bs]
            r_logps = all_logps[bs:]
            del outputs, all_ids, all_mask, all_labels, all_logps

            for j, idx in enumerate(batch_indices):
                if idx < n:
                    local_results[idx] = (c_logps[j].item(), r_logps[j].item())

            if self.accelerator.is_main_process and (bi + 1) % max(1, num_batches // 5) == 0:
                elapsed = time.time() - t0
                print(f"  [REF] {bi+1}/{num_batches} batches, "
                      f"elapsed={elapsed:.1f}s", flush=True)

        # All-gather
        all_results = [None] * self.world_size
        dist.all_gather_object(all_results, local_results)

        ref_chosen = [0.0] * n
        ref_rejected = [0.0] * n
        for rank_dict in all_results:
            for idx, (c, r) in rank_dict.items():
                ref_chosen[idx] = c
                ref_rejected[idx] = r

        self.dataset.ref_chosen_logps = ref_chosen
        self.dataset.ref_rejected_logps = ref_rejected

        if self.accelerator.is_main_process:
            elapsed = time.time() - t0
            print(f"  [REF] Precomputation finished in {elapsed:.1f}s", flush=True)

    # ==========================================================
    # Training step
    # ==========================================================

    def step(self, batch):
        # FIX 4: eval() for dropout consistency with ref
        self.model.eval()

        c_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
        c_mask = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
        c_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
        r_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
        r_mask = batch["rejected_attention_mask"].to(self.device, non_blocking=True)
        r_labels = batch["rejected_labels"].to(self.device, non_blocking=True)
        ref_c = batch["ref_chosen_logps"].to(self.device, non_blocking=True)
        ref_r = batch["ref_rejected_logps"].to(self.device, non_blocking=True)

        if self._micro_step == 0:
            torch.cuda.reset_peak_memory_stats()

        with self.accelerator.accumulate(self.model):
            all_ids = torch.cat([c_ids, r_ids], dim=0)
            all_mask = torch.cat([c_mask, r_mask], dim=0)
            all_labels = torch.cat([c_labels, r_labels], dim=0)
            B = c_ids.size(0)
            del c_ids, r_ids, c_mask, r_mask, c_labels, r_labels

            torch.cuda.empty_cache()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=all_ids, attention_mask=all_mask,
                    use_cache=False, output_router_logits=False, return_dict=True,
                )
            del all_ids, all_mask

            logits = outputs.logits
            del outputs

            if self.args.length_normalise:
                all_logps, all_lengths = compute_per_sequence_logps_with_length(
                    logits, all_labels)
                pc_logps = all_logps[:B]
                pr_logps = all_logps[B:]
                c_lengths = all_lengths[:B]
                r_lengths = all_lengths[B:]
            else:
                all_logps = compute_per_sequence_logps(logits, all_labels)
                pc_logps = all_logps[:B]
                pr_logps = all_logps[B:]
                c_lengths = r_lengths = None
            del logits, all_labels, all_logps

            if self._micro_step == 0 and self.accelerator.is_local_main_process:
                _alloc = torch.cuda.memory_allocated() / 1e9
                _peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[MEM] after fwd+logps: "
                      f"alloc={_alloc:.1f}GB peak={_peak:.1f}GB", flush=True)

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps=pc_logps,
                policy_rejected_logps=pr_logps,
                ref_chosen_logps=ref_c,
                ref_rejected_logps=ref_r,
                beta=self.args.beta,
                label_smoothing=self.args.label_smoothing,
                loss_type=self.args.loss_type,
                length_normalise=self.args.length_normalise,
                chosen_lengths=c_lengths,
                rejected_lengths=r_lengths if self.args.length_normalise else None,
            )
            del pc_logps, pr_logps

            torch.cuda.empty_cache()
            self.accelerator.backward(loss)

            if self._micro_step == 0 and self.accelerator.is_local_main_process:
                _alloc = torch.cuda.memory_allocated() / 1e9
                _peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[MEM] after backward: "
                      f"alloc={_alloc:.1f}GB peak={_peak:.1f}GB", flush=True)

            # FIX 5: Only clip on accumulation boundary
            if self.accelerator.sync_gradients and self.args.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        loss_val = loss.item()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        self._micro_step += 1

        if not (loss_val != loss_val):
            self._micro_loss_accum += loss_val
            self._micro_margin_accum += reward_margin
            self._micro_acc_accum += accuracy
            self._micro_count += 1

        gas = self.args.gradient_accumulation_steps
        if self._micro_step % gas == 0:
            self.global_step += 1
            current_step = self.global_step

            avg_loss = self._micro_loss_accum / max(1, self._micro_count)
            avg_margin = self._micro_margin_accum / max(1, self._micro_count)
            avg_acc = self._micro_acc_accum / max(1, self._micro_count)

            self._epoch_loss_accum += avg_loss
            self._epoch_margin_accum += avg_margin
            self._epoch_acc_accum += avg_acc
            self._epoch_count += 1

            self._micro_loss_accum = 0.0
            self._micro_margin_accum = 0.0
            self._micro_acc_accum = 0.0
            self._micro_count = 0

            step_end = time.time()
            self._step_times.append(step_end)

            if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
                elapsed = step_end - self._train_start_time if self._train_start_time else 0
                run_avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)

                if len(self._step_times) >= 2:
                    window = min(50, len(self._step_times))
                    recent = self._step_times[-1] - self._step_times[-window]
                    steps_per_sec = (window - 1) / max(recent, 1e-6)
                    sec_per_step = 1.0 / max(steps_per_sec, 1e-6)
                else:
                    steps_per_sec = sec_per_step = 0

                eta_str = "?"
                pct_str = ""
                if self.estimated_total_steps > 0:
                    pct = current_step / self.estimated_total_steps * 100
                    pct_str = f" ({pct:.1f}%)"
                    remaining = max(0, self.estimated_total_steps - current_step)
                    if steps_per_sec > 0:
                        eta_sec = remaining / steps_per_sec
                        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

                print(
                    f"  step {current_step}{pct_str} | "
                    f"loss={avg_loss:.4f} (avg={run_avg_loss:.4f}) | "
                    f"margin={avg_margin:.3f} | acc={avg_acc:.3f} | "
                    f"lr={self.opt.param_groups[0]['lr']:.2e} | "
                    f"{sec_per_step:.1f}s/step | "
                    f"elapsed={str(datetime.timedelta(seconds=int(elapsed)))} | "
                    f"ETA={eta_str}",
                    flush=True,
                )

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/avg_loss": run_avg_loss,
                        "train/reward_margin": avg_margin,
                        "train/accuracy": avg_acc,
                        "train/lr": self.opt.param_groups[0]["lr"],
                        "train/epoch": self.global_epoch,
                        "perf/sec_per_step": sec_per_step,
                    }, step=current_step)

            if (self.args.save_every and current_step % self.args.save_every == 0
                    and current_step > 0):
                self.save(os.path.join(self.args.save_dir, f"step-{current_step}"))

    def _should_stop(self):
        return (self.args.steps and self.args.steps > 0
                and self.global_step >= self.args.steps)

    # ==========================================================
    # Save (full model sharded)
    # ==========================================================

    @torch.no_grad()
    def save(self, save_dir):
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
                    "format": "dpo_full_model_sharded",
                    "pattern": "model_sharded.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model": self.args.model,
                    "arch": "Qwen3-Next-80B-A3B",
                    "num_layers": self.args.num_layers,
                    "full_param": True,
                    "freeze_embeddings": self.args.freeze_embeddings,
                    "beta": self.args.beta,
                    "loss_type": self.args.loss_type,
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

    # ==========================================================
    # Training loop
    # ==========================================================

    def train(self):
        self.model.train()
        self._train_start_time = time.time()
        self._step_times = [self._train_start_time]
        self._epoch_loss_accum = 0.0
        self._epoch_margin_accum = 0.0
        self._epoch_acc_accum = 0.0
        self._epoch_count = 0
        self._micro_step = 0
        self._micro_loss_accum = 0.0
        self._micro_margin_accum = 0.0
        self._micro_acc_accum = 0.0
        self._micro_count = 0

        if self.args.epochs and self.args.epochs > 0:
            for ep in range(1, self.args.epochs + 1):
                self.global_epoch = ep
                self.train_sampler.set_epoch(ep)
                epoch_start = time.time()
                epoch_step_start = self.global_step

                self._epoch_loss_accum = 0.0
                self._epoch_margin_accum = 0.0
                self._epoch_acc_accum = 0.0
                self._epoch_count = 0

                if self.accelerator.is_main_process:
                    print(f"\n[EPOCH] start {ep}/{self.args.epochs}", flush=True)

                for batch in self.loader:
                    self.step(batch)
                    if self._should_stop():
                        break

                if self.accelerator.is_main_process:
                    epoch_elapsed = time.time() - epoch_start
                    epoch_steps = self.global_step - epoch_step_start
                    avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)
                    avg_margin = self._epoch_margin_accum / max(1, self._epoch_count)
                    avg_acc = self._epoch_acc_accum / max(1, self._epoch_count)
                    print(
                        f"\n[EPOCH] finished {ep}/{self.args.epochs} "
                        f"| steps={epoch_steps} "
                        f"| avg_loss={avg_loss:.4f} "
                        f"| avg_margin={avg_margin:.3f} "
                        f"| avg_acc={avg_acc:.3f} "
                        f"| time={str(datetime.timedelta(seconds=int(epoch_elapsed)))}",
                        flush=True,
                    )
                if self._should_stop():
                    break

        if not self.args.epochs or self.args.epochs <= 0:
            if self.accelerator.is_main_process:
                print(f"\n[TRAIN] step-only mode (steps={self.args.steps})", flush=True)
            while not self._should_stop():
                self.train_sampler.set_epoch(self.global_step)
                for batch in self.loader:
                    self.step(batch)
                    if self._should_stop():
                        break

        if self.accelerator.is_main_process:
            total_elapsed = time.time() - self._train_start_time
            avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)
            print(
                f"\n[DONE] steps={self.global_step} "
                f"| avg_loss={avg_loss:.4f} "
                f"| total_time={str(datetime.timedelta(seconds=int(total_elapsed)))}",
                flush=True,
            )

        self.save(self.args.save_dir)

        if self.use_wandb:
            import wandb
            wandb.finish()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-Next-80B Full-Param DPO")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./output/qwen3next-fullparam-dpo")
    parser.add_argument("--seed", type=int, default=42)
    # DPO-specific
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--length_normalise", action="store_true")
    # W&B
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    args = parser.parse_args()

    trainer = DPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
