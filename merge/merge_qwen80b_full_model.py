#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO フルパラメータモデルマージ v2 (vLLM 用)

save_pretrained ではなく、ベースモデルの safetensors を
テンプレートとして shard 単位で weight 置換。
→ vLLM が期待する個別 expert フォーマットを維持。

問題の背景:
  モデル内部は stacked expert 形式:
    model.layers.L.mlp.experts.down_proj      → [num_experts, out, in]
    model.layers.L.mlp.experts.gate_up_proj   → [num_experts, 2*intermediate, in]
  ベースモデル safetensors / vLLM は個別 expert:
    model.layers.L.mlp.experts.E.down_proj.weight   → [out, in]
    model.layers.L.mlp.experts.E.gate_proj.weight   → [intermediate, in]
    model.layers.L.mlp.experts.E.up_proj.weight     → [intermediate, in]

処理フロー:
  Phase 1: FSDP rank ファイルをストリーミングマージ → merged_params dict
  Phase 2: ベースモデルの safetensors を shard 単位でロード
           → 各キーの weight を merged_params から置換
           → expert キーは stacked→個別 に変換
           → 同じ shard 名で出力先に保存
  Phase 3: config / tokenizer / カスタムコードをコピー
"""

from __future__ import annotations

import os
import sys
import json
import glob
import gc
import re
import time
import shutil
import argparse
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
    from safetensors.torch import save_file as safetensors_save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ============================================================
# ShardedTensor unpickle hack (FSDP checkpoint 読み込み用)
# ============================================================

HACK_ENABLED = False
HackShardedTensor = None

try:
    import torch.distributed._shard.sharded_tensor.api as st_api

    class HackShardedTensor(torch.Tensor):
        class ProcessGroupState:
            pass
        class Metadata:
            pass
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)
        def __init__(self):
            self._local_shards = []
        def __setstate__(self, state):
            self._local_shards = []
            def is_shard_list(x):
                return isinstance(x, list) and len(x) > 0 and hasattr(x[0], "tensor")
            if isinstance(state, dict):
                for k in ("_local_shards", "local_shards", "_shards", "shards"):
                    v = state.get(k)
                    if is_shard_list(v):
                        self._local_shards = v
                        return
                for v in state.values():
                    if is_shard_list(v):
                        self._local_shards = v
                        return
                    if isinstance(v, dict):
                        for vv in v.values():
                            if is_shard_list(vv):
                                self._local_shards = vv
                                return
            elif isinstance(state, (list, tuple)):
                for it in state:
                    if is_shard_list(it):
                        self._local_shards = it
                        return
                for it in state:
                    if isinstance(it, dict):
                        for v in it.values():
                            if is_shard_list(v):
                                self._local_shards = v
                                return
        def local_shards(self):
            return self._local_shards
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError("HackShardedTensor: not for computation")
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError("HackShardedTensor: not for computation")

    st_api.ShardedTensor = HackShardedTensor
    HACK_ENABLED = True
except Exception:
    pass


def _build_safe_globals():
    safe_list = []
    if HackShardedTensor is not None:
        safe_list.append(HackShardedTensor)
    try:
        import torch.distributed._shard.sharded_tensor.shard as st_shard
        for n in ("Shard", "ShardMetadata", "TensorProperties", "ShardedTensorMetadata"):
            if hasattr(st_shard, n):
                safe_list.append(getattr(st_shard, n))
    except Exception:
        pass
    return safe_list


def load_pt(path, safe_list):
    def _torch_load(**kwargs):
        if hasattr(torch.serialization, "safe_globals") and safe_list:
            with torch.serialization.safe_globals(safe_list):
                return torch.load(path, map_location="cpu", **kwargs)
        return torch.load(path, map_location="cpu", **kwargs)
    for use_mmap in (True, False):
        try:
            return _torch_load(weights_only=False, mmap=use_mmap)
        except TypeError:
            try:
                return _torch_load(mmap=True) if use_mmap else _torch_load()
            except TypeError:
                continue
        except Exception:
            if use_mmap:
                continue
            raise
    return _torch_load()


def clean_key(name):
    name = name.replace("_fsdp_wrapped_module.", "")
    name = name.replace("_checkpoint_wrapped_module.", "")
    name = name.replace("module.", "")
    while name.startswith("model.model."):
        name = "model." + name[len("model.model."):]
    return name


def get_offsets_sizes(meta):
    offsets = sizes = None
    for k in ("shard_offsets", "offsets"):
        if hasattr(meta, k):
            try:
                offsets = [int(x) for x in getattr(meta, k)]
                break
            except Exception:
                pass
    for k in ("shard_sizes", "sizes"):
        if hasattr(meta, k):
            try:
                sizes = [int(x) for x in getattr(meta, k)]
                break
            except Exception:
                pass
    return offsets, sizes


# ============================================================
# Phase 1: FSDP ShardedTensor マージ
# ============================================================

def streaming_merge_sharded(checkpoint_dir, world_size=0, strict=True):
    safe_list = _build_safe_globals()

    meta = {}
    for meta_name in (
        "model_sharded_meta.json",
        "params_meta.json",
        "attn_params_meta.json",
    ):
        meta_path = os.path.join(checkpoint_dir, meta_name)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            print(f"  meta file: {meta_name}", flush=True)
            print(f"  checkpoint meta: {json.dumps(meta, indent=2)}", flush=True)
            break

    if world_size <= 0:
        world_size = meta.get("world_size", 0)

    FILE_PATTERNS = [
        "model_sharded.rank{r}.pt",
        "params.rank{r}.pt",
        "attn_params.rank{r}.pt",
    ]
    GLOB_PATTERNS = [
        "model_sharded.rank*.pt",
        "params.rank*.pt",
        "attn_params.rank*.pt",
        "*rank*.pt",
    ]

    rank_files = []
    if world_size > 0:
        for r in range(world_size):
            found = False
            for pat in FILE_PATTERNS:
                fpath = os.path.join(checkpoint_dir, pat.format(r=r))
                if os.path.exists(fpath):
                    rank_files.append((r, fpath))
                    found = True
                    break
            if not found:
                cands = sorted(glob.glob(os.path.join(checkpoint_dir, f"*rank{r}*.pt")))
                cands = [c for c in cands if "meta" not in c and "full" not in c]
                if cands:
                    rank_files.append((r, cands[0]))
    else:
        for gpat in GLOB_PATTERNS:
            all_f = sorted(glob.glob(os.path.join(checkpoint_dir, gpat)))
            all_f = [f for f in all_f if "meta" not in f and "full" not in f]
            if all_f:
                rank_files = [(i, f) for i, f in enumerate(all_f)]
                world_size = len(all_f)
                break

    if not rank_files:
        raise FileNotFoundError(f"No rank files in {checkpoint_dir}")
    print(f"  rank files: {len(rank_files)} "
          f"(pattern: {os.path.basename(rank_files[0][1])})", flush=True)

    # Pass 1: metadata scan
    print("  [Pass 1] Scanning metadata...", flush=True)
    all_shard_metas: Dict[str, List[Dict]] = {}
    all_full_tensors: Dict[str, torch.Tensor] = {}

    for rank_idx, fpath in rank_files:
        if rank_idx % 10 == 0:
            print(f"    scanning rank {rank_idx}...", flush=True)
        sd = load_pt(fpath, safe_list)
        for raw_name, obj in sd.items():
            name = clean_key(str(raw_name))
            local_shards = None
            if hasattr(obj, "local_shards") and callable(getattr(obj, "local_shards")):
                try:
                    local_shards = obj.local_shards()
                except Exception:
                    pass
            if isinstance(local_shards, list):
                for sh in local_shards:
                    st = getattr(sh, "tensor", None)
                    sm = getattr(sh, "metadata", None)
                    if st is None or sm is None or not torch.is_tensor(st):
                        continue
                    offsets, sizes = get_offsets_sizes(sm)
                    if offsets is None or sizes is None:
                        continue
                    all_shard_metas.setdefault(name, []).append({
                        "offsets": offsets, "sizes": sizes,
                        "dtype": str(st.dtype),
                    })
                continue
            if torch.is_tensor(obj) and name not in all_full_tensors:
                all_full_tensors[name] = obj.detach().contiguous().cpu()
        del sd
        gc.collect()

    print(f"  Found {len(all_shard_metas)} sharded, "
          f"{len(all_full_tensors)} full params", flush=True)

    output_tensors: Dict[str, torch.Tensor] = {}
    for name, metas in all_shard_metas.items():
        dtype_str = metas[0]["dtype"]
        dtype = torch.bfloat16
        if "float32" in dtype_str:
            dtype = torch.float32
        elif "float16" in dtype_str:
            dtype = torch.float16

        ndim = len(metas[0]["offsets"])
        max_dim0 = max(m["offsets"][0] + m["sizes"][0] for m in metas)
        full_shape = [max_dim0] + [metas[0]["sizes"][d] for d in range(1, ndim)]

        if strict:
            sorted_m = sorted(metas, key=lambda m: m["offsets"][0])
            expect = 0
            for m in sorted_m:
                if m["offsets"][0] != expect:
                    raise RuntimeError(
                        f"[STRICT] {name}: gap at dim0 expected={expect} "
                        f"got={m['offsets'][0]}")
                expect += m["sizes"][0]
            if expect != full_shape[0]:
                raise RuntimeError(
                    f"[STRICT] {name}: incomplete {expect}/{full_shape[0]}")

        output_tensors[name] = torch.zeros(full_shape, dtype=dtype)

    # Pass 2: write
    print("  [Pass 2] Streaming write...", flush=True)
    for rank_idx, fpath in rank_files:
        if rank_idx % 10 == 0:
            print(f"    writing rank {rank_idx}...", flush=True)
        sd = load_pt(fpath, safe_list)
        for raw_name, obj in sd.items():
            name = clean_key(str(raw_name))
            local_shards = None
            if hasattr(obj, "local_shards") and callable(getattr(obj, "local_shards")):
                try:
                    local_shards = obj.local_shards()
                except Exception:
                    pass
            if not isinstance(local_shards, list) or name not in output_tensors:
                continue
            out = output_tensors[name]
            for sh in local_shards:
                st = getattr(sh, "tensor", None)
                sm = getattr(sh, "metadata", None)
                if st is None or sm is None:
                    continue
                offsets, sizes = get_offsets_sizes(sm)
                if offsets is None or sizes is None:
                    continue
                sl = tuple(
                    slice(int(o), int(o) + int(s))
                    for o, s in zip(offsets, sizes)
                )
                data = st.detach().contiguous().cpu()
                if data.dtype != out.dtype:
                    data = data.to(out.dtype)
                out[sl].copy_(data.reshape(*sizes))
                del data
        del sd
        gc.collect()

    merged = dict(output_tensors)
    for name, tensor in all_full_tensors.items():
        if name not in merged:
            merged[name] = tensor

    del output_tensors, all_shard_metas, all_full_tensors
    gc.collect()

    print(f"  Merged {len(merged)} params total", flush=True)
    return merged, meta


# ============================================================
# Expert format conversion helpers
# ============================================================

# Pattern: model.layers.L.mlp.experts.E.{down_proj|gate_proj|up_proj}.weight
EXPERT_INDIVIDUAL_RE = re.compile(
    r'^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(down_proj|gate_proj|up_proj)\.weight$'
)


def build_expert_lookup(merged_params):
    """
    merged_params のキーから stacked expert テンソルを検索するための
    lookup dict を構築。

    merged_params のキー例:
      model.layers.0.mlp.experts.down_proj      → [num_experts, out, in]
      model.layers.0.mlp.experts.gate_up_proj   → [num_experts, 2*intermediate, in]

    Returns:
      {
        "model.layers.0.mlp.experts": {
          "down_proj": tensor,      # [E, out, in]
          "gate_up_proj": tensor,   # [E, 2*intermediate, in]
        },
        ...
      }
    """
    lookup = {}
    for key, tensor in merged_params.items():
        # Match: model.layers.L.mlp.experts.{down_proj|gate_up_proj}
        if ".mlp.experts." in key:
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                prefix, attr = parts
                if attr in ("down_proj", "gate_up_proj"):
                    lookup.setdefault(prefix, {})[attr] = tensor
    return lookup


def get_individual_expert_weight(expert_lookup, base_key):
    """
    ベースモデルの個別 expert キーに対応する weight を
    stacked expert テンソルから取り出す。

    base_key: model.layers.L.mlp.experts.E.{down_proj|gate_proj|up_proj}.weight
    Returns: tensor or None
    """
    m = EXPERT_INDIVIDUAL_RE.match(base_key)
    if not m:
        return None

    prefix = m.group(1)     # model.layers.L.mlp.experts
    expert_idx = int(m.group(2))  # E
    proj_type = m.group(3)  # down_proj / gate_proj / up_proj

    expert_data = expert_lookup.get(prefix)
    if expert_data is None:
        return None

    if proj_type == "down_proj":
        stacked = expert_data.get("down_proj")
        if stacked is not None and expert_idx < stacked.shape[0]:
            return stacked[expert_idx].contiguous()

    elif proj_type == "gate_proj":
        stacked = expert_data.get("gate_up_proj")
        if stacked is not None and expert_idx < stacked.shape[0]:
            half = stacked.shape[1] // 2
            return stacked[expert_idx, :half].contiguous()

    elif proj_type == "up_proj":
        stacked = expert_data.get("gate_up_proj")
        if stacked is not None and expert_idx < stacked.shape[0]:
            half = stacked.shape[1] // 2
            return stacked[expert_idx, half:].contiguous()

    return None


def find_merged_key(base_key, merged_params):
    """
    ベースモデルのキーに対応する merged_params のキーを検索。
    non-expert weights 用。
    """
    candidates = [base_key]
    if base_key.startswith("model."):
        candidates.append(base_key[len("model."):])
    else:
        candidates.append("model." + base_key)

    for c in candidates:
        if c in merged_params:
            return c
    return None


# ============================================================
# Phase 2: テンプレートベース保存
# ============================================================

def template_save(base_model_dir, merged_params, out_dir):
    """
    ベースモデルの safetensors を shard 単位でロードし、
    merged_params の weight で置換して保存。
    expert は stacked→個別 に変換。
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors required")

    os.makedirs(out_dir, exist_ok=True)

    # ベースモデルの index.json を読み込み
    index_path = os.path.join(base_model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Not found: {index_path}")

    with open(index_path, "r") as f:
        base_index = json.load(f)

    weight_map = base_index["weight_map"]

    # shard → keys のマッピング
    shard_to_keys: Dict[str, List[str]] = {}
    for key, shard_name in weight_map.items():
        shard_to_keys.setdefault(shard_name, []).append(key)

    # Expert lookup (stacked → individual 変換用)
    expert_lookup = build_expert_lookup(merged_params)
    print(f"  Expert lookup: {len(expert_lookup)} layer prefixes", flush=True)

    # 統計
    total_replaced = 0
    total_expert_replaced = 0
    total_kept = 0
    total_shards = len(shard_to_keys)

    print(f"  Processing {total_shards} shards...", flush=True)

    for shard_idx, (shard_name, keys) in enumerate(sorted(shard_to_keys.items())):
        t0 = time.time()

        # ベースモデルの shard をロード
        shard_path = os.path.join(base_model_dir, shard_name)
        shard_data = safetensors_load_file(shard_path, device="cpu")

        new_shard = OrderedDict()
        replaced_in_shard = 0
        expert_in_shard = 0

        for key in keys:
            base_tensor = shard_data.get(key)
            if base_tensor is None:
                continue

            # 1) Expert 個別キーの場合 → stacked から取り出し
            expert_weight = get_individual_expert_weight(expert_lookup, key)
            if expert_weight is not None:
                if expert_weight.shape != base_tensor.shape:
                    raise RuntimeError(
                        f"Expert weight shape mismatch for {key}: "
                        f"merged {tuple(expert_weight.shape)} vs base {tuple(base_tensor.shape)}"
                        " (stacked→per-expert の分割規約が base 側と食い違っている可能性)"
                    )
                new_shard[key] = expert_weight.to(base_tensor.dtype)
                expert_in_shard += 1
                replaced_in_shard += 1
                continue

            # 2) Non-expert キー → merged_params から直接置換
            merged_key = find_merged_key(key, merged_params)
            if merged_key is not None:
                merged_tensor = merged_params[merged_key]
                if merged_tensor.shape == base_tensor.shape:
                    new_shard[key] = merged_tensor.to(base_tensor.dtype)
                    replaced_in_shard += 1
                else:
                    print(f"    [WARN] shape mismatch {key}: "
                          f"merged={tuple(merged_tensor.shape)} "
                          f"base={tuple(base_tensor.shape)}", flush=True)
                    new_shard[key] = base_tensor
            else:
                # merged_params にない → ベースをそのまま使用
                new_shard[key] = base_tensor

        # 保存
        out_shard_path = os.path.join(out_dir, shard_name)
        safetensors_save_file(new_shard, out_shard_path)

        total_replaced += replaced_in_shard
        total_expert_replaced += expert_in_shard
        total_kept += len(keys) - replaced_in_shard

        del shard_data, new_shard
        gc.collect()

        elapsed = time.time() - t0
        if (shard_idx + 1) % 10 == 0 or shard_idx == 0 or shard_idx == total_shards - 1:
            shard_mb = os.path.getsize(out_shard_path) / (1024 * 1024)
            print(f"    shard {shard_idx+1}/{total_shards}: "
                  f"{shard_name} ({shard_mb:.0f}MB) "
                  f"replaced={replaced_in_shard} (expert={expert_in_shard}) "
                  f"{elapsed:.1f}s", flush=True)

    # index.json をコピー (weight_map は同じ)
    out_index_path = os.path.join(out_dir, "model.safetensors.index.json")
    shutil.copy2(index_path, out_index_path)

    total_keys = sum(len(v) for v in shard_to_keys.values())
    print(f"\n  Template save complete:", flush=True)
    print(f"    Total keys    : {total_keys}", flush=True)
    print(f"    Replaced      : {total_replaced} "
          f"(expert={total_expert_replaced}, "
          f"other={total_replaced - total_expert_replaced})", flush=True)
    print(f"    Kept from base: {total_kept}", flush=True)

    return total_replaced


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="DPO full-param model merge v2 (template-based save)")
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_layers", type=int, default=48)
    ap.add_argument("--world_size", type=int, default=0)
    ap.add_argument("--no_strict", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print("  DPO Full-Param Model Merge v2 (template-based)")
    print("=" * 64)
    print(f"  base_model     : {args.base_model}")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  out_dir        : {args.out_dir}")
    print(f"  num_layers     : {args.num_layers}")
    print(f"  world_size     : {args.world_size} (0=auto)")
    print(f"  hack_enabled   : {HACK_ENABLED}")
    print("=" * 64, flush=True)

    total_start = time.time()

    # ================================================================
    # Phase 1: FSDP sharded checkpoint マージ
    # ================================================================
    print("\n" + "=" * 64)
    print("  Phase 1: Streaming merge of FSDP sharded checkpoints")
    print("=" * 64, flush=True)

    merged_params, ckpt_meta = streaming_merge_sharded(
        args.checkpoint_dir,
        world_size=args.world_size,
        strict=not args.no_strict,
    )

    # サマリー
    total_elements = sum(t.numel() for t in merged_params.values())
    print(f"\n  Merged: {len(merged_params)} keys, "
          f"{total_elements:,} params ({total_elements/1e9:.2f}B)", flush=True)

    # Expert キーの確認
    expert_keys = [k for k in merged_params if ".mlp.experts." in k]
    non_expert_keys = [k for k in merged_params if ".mlp.experts." not in k]
    print(f"  Expert stacked keys : {len(expert_keys)}", flush=True)
    print(f"  Non-expert keys     : {len(non_expert_keys)}", flush=True)
    for k in expert_keys[:6]:
        t = merged_params[k]
        print(f"    {k}: {tuple(t.shape)} {t.dtype}", flush=True)
    if len(expert_keys) > 6:
        print(f"    ... and {len(expert_keys) - 6} more", flush=True)

    # ================================================================
    # Phase 2: テンプレートベース保存
    # ================================================================
    print("\n" + "=" * 64)
    print(f"  Phase 2: Template-based save → {args.out_dir}")
    print("=" * 64, flush=True)

    n_replaced = template_save(args.base_model, merged_params, args.out_dir)

    del merged_params
    gc.collect()

    # ================================================================
    # Phase 3: config / tokenizer / カスタムコード
    # ================================================================
    print("\n" + "=" * 64)
    print("  Phase 3: Copy config, tokenizer, custom code")
    print("=" * 64, flush=True)

    copy_targets = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ]
    for fname in copy_targets:
        src = os.path.join(args.base_model, fname)
        dst = os.path.join(args.out_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"    copied {fname}", flush=True)

    for fname in os.listdir(args.base_model):
        if fname.endswith(".py"):
            src = os.path.join(args.base_model, fname)
            dst = os.path.join(args.out_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"    copied {fname}", flush=True)

    # ================================================================
    # サマリー
    # ================================================================
    total_elapsed = time.time() - total_start
    total_size = sum(
        os.path.getsize(os.path.join(args.out_dir, f))
        for f in os.listdir(args.out_dir)
        if os.path.isfile(os.path.join(args.out_dir, f))
    )

    print("\n" + "=" * 64)
    print(f"  ✅ DPO マージ完了!")
    print(f"  出力先       : {args.out_dir}")
    print(f"  出力サイズ   : {total_size / (1024**3):.1f} GB")
    print(f"  置換パラメータ: {n_replaced}")
    print(f"  合計時間     : {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")
    print("=" * 64)
    print(f"\n  vllm serve {args.out_dir} --trust-remote-code")


if __name__ == "__main__":
    main()
