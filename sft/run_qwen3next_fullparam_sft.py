#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-parameter SFT for Qwen3-Next-80B-A3B (FSDP, BF16).

Adapted from Qwen3.5-397B partial-layer SFT.

Supports:
  - Full fine-tuning of all parameters (default)
  - Optional: freeze embeddings (--freeze_embeddings)
  - Sequence packing (--packing) for GPU utilization
  - Gradient accumulation (--gradient_accumulation_steps)
  - W&B logging (--wandb_project)

FIX 5: clip_grad_norm_ guarded by sync_gradients (accumulation safety)
FIX 6: GradientAccumulationPlugin with sync_each_batch=True (FSDP OOM)

Model: Qwen3-Next-80B-A3B-Instruct
  - 48 layers, hybrid attention (full_attention_interval=4)
    → 12 FullAttn + 36 Gated DeltaNet
  - 512 MoE experts (10 routed + 1 shared)
  - BF16 native, ~80B total / 3B active per token
  - No vision encoder (text-only)

Memory (5 nodes = 40 GPU, FSDP FULL_SHARD):
  Weights  : 160GB / 40 = 4GB/GPU
  Adam     : 640GB / 40 = 16GB/GPU
  Grads    : 160GB / 40 = 4GB/GPU
  Total    : ~24GB/GPU (H100 80GB → comfortable)

Loading strategy:
  - rank 0: from_pretrained() loads BF16 weights to CPU
  - rank != 0: meta context → zero CPU RAM
  - FSDP sync_module_states=True broadcasts from rank 0
"""

import os
import json
import math
import random
import argparse
import datetime
import time
from typing import List, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
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
# Dataset: On-the-fly ChatML tokenization
# ============================================================

def parse_chatml(text):
    """Parse ChatML format into list of (role, content)."""
    messages = []
    for block in text.split("<|im_start|>"):
        block = block.strip()
        if not block:
            continue
        block = block.replace("<|im_end|>", "").strip()
        if "\n" in block:
            role, content = block.split("\n", 1)
            messages.append({"role": role.strip(), "content": content.strip()})
        else:
            messages.append({"role": block.strip(), "content": ""})
    return messages


class OnTheFlyTokenizedChatMLJsonl(IterableDataset):
    """Reads .jsonl with ChatML text, tokenizes on-the-fly."""

    def __init__(self, path, tok, seq_len, rank=0, world_size=1):
        self.path = path
        self.tok = tok
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

        with open(path, "r", encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]
        self.total = len(self.lines)

    def _tokenize_line(self, line):
        """Tokenize a single JSONL line with manual assistant-turn masking."""
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None

        text = obj.get("text", None)
        messages = obj.get("messages", None)

        if messages:
            pass
        elif text:
            messages = parse_chatml(text)
        else:
            return None

        if not messages:
            return None

        # Get formatted text string (no tokenization yet)
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

        # Tokenize the full formatted text
        enc = self.tok(formatted, add_special_tokens=False)
        input_ids = enc["input_ids"]
        if not input_ids:
            return None

        # Build assistant mask from string positions
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

        # Sanity: if no assistant tokens found, train on everything
        if all(l == -100 for l in labels):
            labels = list(input_ids)

        return {"input_ids": input_ids, "labels": labels}

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            per_worker = len(self.lines) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.lines)
            lines = self.lines[start:end]
        else:
            lines = self.lines

        for idx, line in enumerate(lines):
            if idx % self.world_size != self.rank:
                continue
            result = self._tokenize_line(line)
            if result is None:
                continue
            ids = result["input_ids"][:self.seq_len]
            labs = result["labels"][:self.seq_len]
            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labs, dtype=torch.long),
                "length": len(ids),
            }


class PackedDataset(IterableDataset):
    """Packs multiple samples into fixed-length sequences."""

    def __init__(self, base_dataset, seq_len, batch_size, pad_token_id):
        self.base = base_dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_id = pad_token_id

    def _pack_iter(self):
        buf_ids, buf_labels = [], []
        for sample in self.base:
            ids = sample["input_ids"].tolist()
            labs = sample["labels"].tolist()
            buf_ids.extend(ids)
            buf_labels.extend(labs)
            while len(buf_ids) >= self.seq_len:
                chunk_ids = buf_ids[:self.seq_len]
                chunk_labels = buf_labels[:self.seq_len]
                buf_ids = buf_ids[self.seq_len:]
                buf_labels = buf_labels[self.seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk_ids, dtype=torch.long),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                    "labels": torch.tensor(chunk_labels, dtype=torch.long),
                }
        if buf_ids:
            pad_len = self.seq_len - len(buf_ids)
            buf_ids.extend([self.pad_id] * pad_len)
            buf_labels.extend([-100] * pad_len)
            yield {
                "input_ids": torch.tensor(buf_ids[:self.seq_len], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [1] * (self.seq_len - pad_len) + [0] * pad_len, dtype=torch.long),
                "labels": torch.tensor(buf_labels[:self.seq_len], dtype=torch.long),
            }

    def __iter__(self):
        batch = []
        for packed in self._pack_iter():
            batch.append(packed)
            if len(batch) >= self.batch_size:
                yield {
                    "input_ids": torch.stack([b["input_ids"] for b in batch]),
                    "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                    "labels": torch.stack([b["labels"] for b in batch]),
                }
                batch = []
        if batch:
            yield {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
            }


# ============================================================
# Layer class discovery for FSDP wrapping
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
# Trainer
# ============================================================

class Trainer:
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

        if wrap_class is None:
            with torch.device("meta"):
                try:
                    temp_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                    wrap_class = discover_layer_class_from_model(temp_model, args.num_layers)
                    del temp_model
                except Exception:
                    pass

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
                args.model,
                config=config,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
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
                        args.model,
                        config=config,
                        device_map=None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
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

        # 4) Freeze control
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

            # Hybrid attention info
            tc = getattr(config, "text_config", config)
            fai = getattr(tc, "full_attention_interval", None)
            if fai:
                n_full = len([i for i in range(args.num_layers) if i % fai == 0])
                n_delta = args.num_layers - n_full
                print(f"[INFO] Hybrid attention: interval={fai} → "
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
            weight_decay=args.weight_decay,
            foreach=False,
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
            print(f"[DATA] packing: {args.packing}", flush=True)

        base_ds = OnTheFlyTokenizedChatMLJsonl(
            args.dataset, tok=self.tok, seq_len=args.seq_len,
            rank=self.rank, world_size=self.world_size,
        )

        if args.packing:
            ds = PackedDataset(base_ds, args.seq_len, args.batch_size, self.pad_token_id)
            self.loader = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=True)

            avg_tokens = 500
            if self.rank == 0:
                sample_n = min(50, len(base_ds.lines))
                sample_lens = []
                for i in range(sample_n):
                    r = base_ds._tokenize_line(base_ds.lines[i])
                    if r is not None:
                        sample_lens.append(len(r["input_ids"]))
                        if len(sample_lens) == 1:
                            ids = r["input_ids"]
                            labs = r["labels"]
                            n_train = sum(1 for l in labs if l != -100)
                            n_mask = sum(1 for l in labs if l == -100)
                            print(f"[DATA] sample 0: {len(ids)} tokens, "
                                  f"{n_train} trainable, {n_mask} masked", flush=True)
                if sample_lens:
                    avg_tokens = sum(sample_lens) // len(sample_lens)
                    print(f"[DATA] sampled {len(sample_lens)} lines → avg {avg_tokens} tokens/sample",
                          flush=True)

            rank_samples = base_ds.total // self.world_size
            rank_tokens = rank_samples * avg_tokens
            packed_seqs = max(1, rank_tokens // args.seq_len)
            if self.accelerator.is_main_process:
                print(f"[DATA] packing estimate: ~{rank_tokens:,} tokens/rank, "
                      f"~{packed_seqs} packed_seqs/rank", flush=True)
        else:
            def collate_fn(batch):
                max_len = max(b["length"] for b in batch)
                max_len = min(max_len, args.seq_len)
                ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
                mask = torch.zeros(len(batch), max_len, dtype=torch.long)
                labs = torch.full((len(batch), max_len), -100, dtype=torch.long)
                for i, b in enumerate(batch):
                    slen = min(b["length"], max_len)
                    ids[i, :slen] = b["input_ids"][:slen]
                    mask[i, :slen] = 1
                    labs[i, :slen] = b["labels"][:slen]
                return {"input_ids": ids, "attention_mask": mask, "labels": labs}

            self.loader = DataLoader(
                base_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                num_workers=0, pin_memory=True, drop_last=False,
            )

        # Estimate steps
        gas = args.gradient_accumulation_steps
        eff_batch = args.batch_size * gas * self.world_size
        if args.packing:
            total_tokens_est = base_ds.total * avg_tokens
            total_packed = max(1, total_tokens_est // args.seq_len)
            micro_per_epoch = max(1, total_packed // (args.batch_size * self.world_size))
        else:
            lines_per_rank = base_ds.total // self.world_size
            micro_per_epoch = max(1, lines_per_rank // args.batch_size)
        optim_per_epoch = micro_per_epoch // gas
        total_steps = optim_per_epoch * (args.epochs or 1)
        self.estimated_total_steps = max(1, total_steps)

        if self.accelerator.is_main_process:
            print(f"[DATA] total_lines={base_ds.total}, "
                  f"micro_batches/epoch≈{micro_per_epoch}, "
                  f"optim_steps/epoch≈{optim_per_epoch}, "
                  f"estimated_total_steps≈{total_steps}, "
                  f"effective_batch_size={eff_batch}", flush=True)

        # Summary
        if self.accelerator.is_main_process:
            print("=" * 56)
            print("  Qwen3-Next-80B-A3B Full-Param SFT")
            print(f"  model          : {args.model}")
            print(f"  num_layers     : {args.num_layers}")
            print(f"  freeze_embed   : {args.freeze_embeddings}")
            print(f"  dataset        : {args.dataset}")
            print(f"  packing        : {args.packing}")
            print(f"  batch_size     : {args.batch_size} (micro)")
            print(f"  grad_accum     : {gas}")
            print(f"  eff_batch      : {eff_batch}")
            print(f"  seq_len        : {args.seq_len}")
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

    def step(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        self.model.train()

        with self.accelerator.accumulate(self.model):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                output_router_logits=False,
                return_dict=True,
            )
            loss = outputs.loss

            self.accelerator.backward(loss)

            # FIX 5: Only clip on accumulation boundary
            if self.accelerator.sync_gradients and self.args.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        loss_val = loss.item()
        self._micro_step += 1

        if not (loss_val != loss_val):  # not NaN
            self._micro_loss_accum += loss_val
            self._micro_loss_count += 1

        gas = self.args.gradient_accumulation_steps
        if self._micro_step % gas == 0:
            self.global_step += 1
            current_step = self.global_step

            accum_avg_loss = self._micro_loss_accum / max(1, self._micro_loss_count)
            self._micro_loss_accum = 0.0
            self._micro_loss_count = 0
            self._loss_accum += accum_avg_loss
            self._loss_count += 1

            step_end = time.time()
            if self._train_start_time is not None:
                self._step_times.append(step_end)

            if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
                elapsed = step_end - self._train_start_time if self._train_start_time else 0
                avg_loss = self._loss_accum / max(1, self._loss_count)

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
                    f"loss={accum_avg_loss:.4f} (avg={avg_loss:.4f}) | "
                    f"lr={self.opt.param_groups[0]['lr']:.2e} | "
                    f"{sec_per_step:.1f}s/step | "
                    f"elapsed={str(datetime.timedelta(seconds=int(elapsed)))} | "
                    f"ETA={eta_str}",
                    flush=True,
                )

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": accum_avg_loss,
                        "train/avg_loss": avg_loss,
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
                    "format": "full_model_sharded",
                    "pattern": "model_sharded.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model": self.args.model,
                    "arch": "Qwen3-Next-80B-A3B",
                    "num_layers": self.args.num_layers,
                    "full_param": True,
                    "freeze_embeddings": self.args.freeze_embeddings,
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

    def train(self):
        self.model.train()
        self._train_start_time = time.time()
        self._step_times = [self._train_start_time]
        self._loss_accum = 0.0
        self._loss_count = 0
        self._micro_step = 0
        self._micro_loss_accum = 0.0
        self._micro_loss_count = 0

        if self.args.epochs and self.args.epochs > 0:
            for ep in range(1, self.args.epochs + 1):
                self.global_epoch = ep
                epoch_start = time.time()
                epoch_step_start = self.global_step
                if self.accelerator.is_main_process:
                    print(f"\n[EPOCH] start {ep}/{self.args.epochs}", flush=True)
                # Ranks can yield different numbers of batches per epoch
                # (per-rank dataset sharding + packing remainders).  If one
                # rank keeps calling step() (FSDP collectives) after another
                # rank's loader is exhausted, training deadlocks.  Align all
                # ranks to the global minimum batch count: before every step,
                # all_reduce(MIN) a "has next batch" flag and stop as soon as
                # any rank runs out.
                loader_iter = iter(self.loader)
                while True:
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        batch = None
                    if dist.is_available() and dist.is_initialized():
                        has_batch = torch.tensor(
                            [0 if batch is None else 1],
                            dtype=torch.long,
                            device=self.device,
                        )
                        dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)
                        if int(has_batch.item()) == 0:
                            if batch is not None and self.rank == 0:
                                print(
                                    "[EPOCH] another rank exhausted its loader "
                                    "— truncating epoch to the global minimum "
                                    "batch count",
                                    flush=True,
                                )
                            break
                    elif batch is None:
                        break
                    self.step(batch)
                    if self._should_stop():
                        break
                if self.accelerator.is_main_process:
                    epoch_elapsed = time.time() - epoch_start
                    epoch_steps = self.global_step - epoch_step_start
                    avg_loss = self._loss_accum / max(1, self._loss_count)
                    print(
                        f"\n[EPOCH] finished {ep}/{self.args.epochs} "
                        f"| steps={epoch_steps} "
                        f"| avg_loss={avg_loss:.4f} "
                        f"| time={str(datetime.timedelta(seconds=int(epoch_elapsed)))}",
                        flush=True,
                    )
                if self._should_stop():
                    break

        if not self.args.epochs or self.args.epochs <= 0:
            if self.accelerator.is_main_process:
                print(f"\n[TRAIN] step-only mode (steps={self.args.steps})", flush=True)
            while not self._should_stop():
                for batch in self.loader:
                    self.step(batch)
                    if self._should_stop():
                        break

        # Final save
        if self.accelerator.is_main_process:
            total_elapsed = time.time() - self._train_start_time
            avg_loss = self._loss_accum / max(1, self._loss_count)
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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Next-80B Full-Param SFT")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./output/qwen3next-fullparam-sft")
    parser.add_argument("--seed", type=int, default=42)
    # W&B
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
