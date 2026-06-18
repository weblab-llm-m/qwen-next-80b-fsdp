# Qwen3-Next-80B-A3B-FSDP

Full-parameter training framework for **Qwen3-Next-80B-A3B** using FSDP (Fully Sharded Data Parallel). Provides three training pipelines — SFT / DPO / GRPO — plus checkpoint merging back into a vLLM-compatible model. During GRPO, trained weights are streamed to a vLLM inference server in real time over NCCL. Designed for SLURM + H100 multi-node clusters.

## Model Overview

| Item | Value |
|---|---|
| Model | Qwen3-Next-80B-A3B |
| Total Parameters | 80B (3B active/token) |
| Layers | 48 (hidden_size=2048, head_dim=256) |
| Attention | Hybrid: 12 Full Attention + 36 Gated DeltaNet (full_attention_interval=4) |
| MoE | 512 routed experts (10 active/token) + 1 shared expert/layer |
| Precision | BF16 native |
| Vocab | 151,936 |

> `torch.compile` is not supported. Use `--enforce-eager` for vLLM and `TORCHDYNAMO_DISABLE=1` for the trainer. Qwen3-Next also does **not** support vLLM prefix caching, and the Gated DeltaNet (GDN) FlashInfer kernels are JIT-compiled on first use (40–60+ min on a cold cache).

## Features

- **Full-parameter FSDP** — FULL_SHARD across all MoE experts and attention layers (optional embedding/lm_head freeze)
- **Three training pipelines** — SFT, DPO, GRPO with LLM-as-judge reward (benchmark-style 0–9 scoring)
- **FSDP-vLLM weight sync** — Real-time NCCL weight broadcast from trainer to vLLM during GRPO, one FSDP unit at a time
- **Automatic vLLM recovery** — Server watchdog + auto-restart loop, client-side NCCL reconnection and weight resync
- **MoE-aware DPO** — Concat forward pass for routing consistency between chosen/rejected
- **Checkpointed cross-entropy** — Memory-efficient logits softmax to prevent OOM
- **Sequence packing** — Pack multiple short samples per fixed-length sequence (SFT)
- **Judge score caching** — Per-rank JSONL cache keyed by SHA-1 of (prompt, candidate, reference+truth)
- **W&B integration** — Optional Weights & Biases experiment tracking (metrics + sample tables)

## Directory Structure

```
.
├── env.sh                                  # Unified conda environment setup
├── sft/                                    # SFT (Supervised Fine-Tuning)
│   ├── run_qwen3next_fullparam_sft.py      # Trainer main script
│   └── run_qwen3next_fullparam_sft.sh      # SLURM job script (5 nodes)
├── dpo/                                    # DPO (Direct Preference Optimization)
│   ├── run_qwen3next_fullparam_dpo.py      # Trainer main script
│   └── run_qwen3next_fullparam_dpo.sh      # SLURM job script (5 nodes)
├── grpo/                                   # GRPO (Reinforcement Learning)
│   ├── run_qwen3next_fullparam_grpo.py     # Trainer main script
│   ├── run_qwen3next_fullparam_grpo.sh     # SLURM job script (1 vLLM + 4 trainer nodes)
│   ├── fsdp_vllm_client.py                 # FSDP-aware vLLM client (NCCL weight sync + recovery)
│   ├── vllm_trl_serve.py                   # vLLM + TRL weight sync server (FastAPI)
│   └── triton_worker_extension.py          # Triton allocator fix for vLLM workers
└── merge/                                  # FSDP sharded checkpoint merge
    ├── merge_qwen80b_full_model.py         # Merge script (shard → vLLM format)
    └── merge_qwen80b_full_model.sh         # SLURM job script (CPU-only)
```

## Prerequisites

- **GPU**: NVIDIA H100 80GB multi-node cluster (5 nodes × 8 GPUs for the reference configs)
- **CUDA**: 12.4 driver (`module load cuda/12.4`)
- **Conda**: miniconda / conda (Python 3.11)
- **SLURM**: Job scheduler
- **NCCL**: 2.22.3 (`module load nccl/2.22.3`)
- **Network**: High-speed interconnect (InfiniBand / RoCE) for NCCL

## Setup

### 1. Create the Conda Environment

```bash
bash env.sh
conda activate qwen3next
```

A single environment (`qwen3next`) serves both the vLLM server and the FSDP trainer (Qwen3-Next runs on transformers 4.57+, so no split environments are needed). The script performs:

1. Create conda env `qwen3next` (Python 3.11; prompts before recreating an existing env)
2. Install PyTorch (CUDA 12.4 wheels)
3. Install vLLM (stable; falls back to nightly from `wheels.vllm.ai` — vLLM >= 0.10.2 is required for Qwen3-Next / GDN kernels)
4. Install transformers with Qwen3-Next support (latest stable; falls back to `git+.../transformers@main` if `Qwen3NextForCausalLM` is missing)
5. Install TRL (provides `WeightSyncWorkerExtension`)
6. Install accelerate, requests, fastapi, uvicorn, pydantic, wandb
7. Create a `vllm_ascend` stub package (TRL import compatibility for `WeightSyncWorkerExtension`)
8. Final verification of all imports

### 2. Prepare the Data

**SFT / GRPO** — ChatML-messages JSONL (one JSON object per line):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

- SFT also accepts a raw ChatML string in a `text` field (parsed via `<|im_start|>` / `<|im_end|>` markers). Loss is computed on assistant turns only (non-assistant tokens are label-masked).
- GRPO uses the first `user` message as the prompt and the first `assistant` message as the reference answer. An optional `metadata.original_truth` field is passed to the judge as the ground truth:

```json
{"messages": [...], "metadata": {"original_truth": "..."}}
```

**DPO** — prompt / chosen / rejected JSONL:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

### 3. Configure Paths

Edit each `.sh` file to match your environment:

```bash
BASE_MODEL="/path/to/base_model"
DATASET="/path/to/dataset.jsonl"
```

Also adjust the SLURM settings (`--partition`, `--nodelist`) and the network interface (`TARGET_IFNAME`, default `eth0`).

**GRPO additional setup:**

```bash
export OPENROUTER_API_KEY="your-api-key"   # Required for the LLM judge (or JUDGE_API_KEY)
```

---

## 1. Supervised Fine-Tuning (SFT)

Full-parameter supervised fine-tuning on ChatML-formatted JSONL data. Tokenization is done on the fly with assistant-turn-only label masking; optional sequence packing concatenates samples into fixed-length `seq_len` chunks to eliminate padding waste.

- **Node config**: 5 nodes × 8 GPUs = 40 GPUs (FSDP FULL_SHARD)
- **Memory**: ~24 GB/GPU (H100 80GB)
- **Features**: Sequence packing, assistant-turn-only training, optional embedding freeze, gradient checkpointing

### Launch

```bash
sbatch sft/run_qwen3next_fullparam_sft.sh
```

### Basic Usage

```bash
accelerate launch \
    --multi_gpu \
    --num_machines 5 \
    --num_processes 40 \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --mixed_precision no \
    sft/run_qwen3next_fullparam_sft.py \
    --model /path/to/Qwen3-Next-80B-A3B-Instruct \
    --num_layers 48 \
    --dataset /path/to/train_sft.jsonl \
    --seq_len 2048 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --packing \
    --epochs 1 \
    --lr 1e-5 \
    --save_every 50 \
    --wandb_project qwen3next-sft \
    --save_dir ./output/qwen3next-fullparam-sft
```

> `--mixed_precision no` is intentional: BF16 is applied through the FSDP `MixedPrecision` policy (param/reduce/buffer dtype = bfloat16), not through Accelerate AMP.

### Full Argument Reference

#### Model & Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Path to base model |
| `--num_layers` | int | 48 | Number of decoder layers (used for FSDP wrap-class discovery) |
| `--freeze_embeddings` | flag | off | Freeze `embed_tokens` and `lm_head` |
| `--dataset` | str | **required** | Path to ChatML JSONL training data |
| `--seq_len` | int | 4096 | Maximum sequence length (also the packed chunk length) |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--batch_size` | int | 1 | Micro-batch size per GPU |
| `--gradient_accumulation_steps` | int | 4 | Gradient accumulation steps (`sync_each_batch=True` for FSDP OOM safety) |
| `--packing` | flag | off | Enable sequence packing (multiple samples per `seq_len` chunk) |
| `--epochs` | int | 1 | Number of epochs (`<=0` switches to step-only mode) |
| `--steps` | int | 0 | Maximum optimizer steps (0 = unlimited; acts as a cap in epoch mode) |
| `--lr` | float | 1e-5 | Learning rate (AdamW, betas=(0.9, 0.98)) |
| `--weight_decay` | float | 0.01 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold (0 to disable; applied only on accumulation boundaries) |
| `--seed` | int | 42 | Random seed |

#### Save & Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--log_every` | int | 1 | Log output interval (optimizer steps) |
| `--save_every` | int | 0 | Save intermediate checkpoint every N steps into `save_dir/step-N` (0 = final save only) |
| `--save_dir` | str | ./output/qwen3next-fullparam-sft | Checkpoint save directory |
| `--wandb_project` | str | "" | W&B project name (empty string disables W&B) |
| `--wandb_run_name` | str | "" | W&B run name |

**Checkpoint format**: each rank writes `model_sharded.rank{N}.pt` (FSDP `SHARDED_STATE_DICT`, CPU-offloaded) plus a `model_sharded_meta.json` describing the format, world size, and step count. Use the [merge tool](#5-model-merging) to convert to a vLLM-loadable model.

---

## 2. Direct Preference Optimization (DPO)

Full-parameter DPO on {prompt, chosen, rejected} pairs. Reference log-probabilities are precomputed with the frozen initial policy before training starts (no second model copy in memory). Contains several MoE-specific correctness fixes:

- **FIX 1** — Collator pads chosen and rejected to the *same* max length (`torch.cat` compatibility)
- **FIX 2** — Reference precompute uses a *concatenated* forward pass (chosen+rejected in one batch) for MoE routing consistency
- **FIX 3** — Reference batch size forced to 1 (batch=2 concatenated, exactly matching the training forward)
- **FIX 4** — `model.eval()` during training steps (dropout consistency with the reference pass)
- **FIX 5** — `clip_grad_norm_` guarded by `sync_gradients` (accumulation safety)
- **FIX 6** — `GradientAccumulationPlugin(sync_each_batch=True)` (FSDP grad-buffer OOM mitigation)
- **FIX 7** — Checkpointed cross-entropy (avoids materializing the `(B*T, V)` float32 softmax)

- **Node config**: 5 nodes × 8 GPUs = 40 GPUs (FSDP FULL_SHARD)
- **Memory**: ~30 GB/GPU (H100 80GB) — weights+grads+Adam ~24 GB + concat forward activations ~4–6 GB

### Processing Flow

1. Tokenize all pairs with assistant-turn label masking (pairs with no trainable tokens are dropped)
2. Precompute reference log-probs with the initial policy (concat forward, ref_bs=1, all-gathered across ranks)
3. Train: concat forward → per-sequence log-probs (checkpointed CE) → DPO loss → backward → clip → AdamW step
4. Save FSDP sharded checkpoint (`model_sharded.rank*.pt` + meta JSON)

### Launch

```bash
sbatch dpo/run_qwen3next_fullparam_dpo.sh
```

### Basic Usage

```bash
accelerate launch \
    --multi_gpu \
    --num_machines 5 \
    --num_processes 40 \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --mixed_precision no \
    dpo/run_qwen3next_fullparam_dpo.py \
    --model /path/to/Qwen3-Next-80B-A3B-Instruct \
    --num_layers 48 \
    --dataset /path/to/train_dpo.jsonl \
    --seq_len 4096 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --epochs 3 \
    --lr 5e-6 \
    --beta 0.1 \
    --loss_type sigmoid \
    --save_every 50 \
    --wandb_project qwen3next-dpo \
    --save_dir ./output/qwen3next-fullparam-dpo
```

### Full Argument Reference

#### Model & Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Path to base model |
| `--num_layers` | int | 48 | Number of decoder layers |
| `--freeze_embeddings` | flag | off | Freeze `embed_tokens` and `lm_head` |
| `--dataset` | str | **required** | Path to DPO JSONL (prompt/chosen/rejected) |
| `--seq_len` | int | 4096 | Maximum sequence length |

#### DPO Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--beta` | float | 0.1 | DPO temperature β (controls deviation from the reference policy) |
| `--loss_type` | str | sigmoid | Loss variant (`sigmoid` / `hinge` / `ipo`) |
| `--label_smoothing` | float | 0.0 | Label smoothing factor (sigmoid loss only) |
| `--length_normalise` | flag | off | Normalize log-probs by sequence length before computing rewards |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--batch_size` | int | 1 | Micro-batch size per rank (per pair; concat forward runs batch=2) |
| `--gradient_accumulation_steps` | int | 4 | Gradient accumulation steps |
| `--epochs` | int | 3 | Number of epochs (`<=0` switches to step-only mode) |
| `--steps` | int | 0 | Maximum optimizer steps (0 = unlimited) |
| `--lr` | float | 5e-6 | Learning rate (AdamW, betas=(0.9, 0.98)) |
| `--weight_decay` | float | 0.01 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold (0 to disable) |
| `--seed` | int | 42 | Random seed (also used by the `DistributedSampler`) |

#### Save & Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--log_every` | int | 1 | Log output interval (steps; logs loss, reward margin, accuracy) |
| `--save_every` | int | 0 | Save checkpoint every N steps (0 = final save only) |
| `--save_dir` | str | ./output/qwen3next-fullparam-dpo | Checkpoint save directory |
| `--wandb_project` | str | "" | W&B project name (empty disables) |
| `--wandb_run_name` | str | "" | W&B run name |

---

## 3. GRPO (Reinforcement Learning)

Full-parameter GRPO: candidate responses are generated by a dedicated vLLM server node, scored by an external LLM-as-judge (0–9 integer score, benchmark style with `original_truth`), and the policy is updated with a group-normalized policy gradient. Trainer weights are streamed to vLLM over NCCL after every optimizer step.

- **Node config**: 1 node (vLLM, TP=8, mp backend — no Ray) + 4 nodes (FSDP trainer, 32 GPUs) = 5 nodes
- **Memory**: Trainer ~30 GB/GPU (H100 80GB)
- **Features**: NCCL streaming weight sync, watchdog-based automatic vLLM recovery, OpenRouter/OpenAI judge with per-rank score caching, overlong-prompt filtering

### GRPO Architecture

```
┌──────────────┐    HTTP /generate     ┌──────────────────────┐
│  Trainer     │ ─────────────────────→ │  vLLM Server         │
│  (4 nodes,   │                       │  (1 node, TP=8)      │
│   32 GPUs,   │  NCCL weight broadcast │                      │
│   FSDP)      │ ─────────────────────→ │  WeightSync          │
│              │                       │  WorkerExtension     │
└──────┬───────┘                       └──────────────────────┘
       │
       │  HTTP (OpenRouter / OpenAI API)
       ↓
┌──────────────┐
│  LLM Judge   │
│  (0–9 score) │
└──────────────┘
```

### Processing Flow (per step)

1. **Gather** — all ranks gather their prompts/references/truths to rank 0 (`all_gather_object`)
2. **Generate** — rank 0 requests `grpo_group_size` candidates per prompt from the vLLM server(s) via HTTP `/generate` (multi-server requests are round-robin sharded and issued in parallel)
3. **Judge** — rank 0 scores every (prompt, candidate) with the external judge in a thread pool (`judge_concurrency` workers); cached scores are reused, failures become NaN
4. **Broadcast** — candidates + scores are broadcast to all ranks; each rank slices out its own portion
5. **Update** — per-group advantage `(score - group_mean) / (group_std + 1e-6)` (NaN-masked; groups with <2 valid samples get advantage 0), scaled by `reward_scale` and clamped to `±reward_clip`; loss = `-(advantage · mean token log-prob)`; backward → clip → AdamW step
6. **Sync** — every `vllm_update_every` steps, all ranks stream the updated weights to vLLM (per-FSDP-unit `summon_full_params` → NCCL broadcast from rank 0), then the prefix cache is reset

An EMA reward baseline (`ema_alpha`) is tracked and logged for monitoring. Judge scores are cached in a per-rank file (the `--reward_cache` path is automatically suffixed `.rank{N}`).

**Recovery**: if generation fails (vLLM hang/crash), the client kills the server via `/force_exit`, waits for the shell watchdog's auto-restart, verifies stability, re-establishes NCCL, and raises `VLLMRecoveryNeeded`; the trainer then resyncs weights on all ranks and retries the generation once. If the retry also fails, the step is skipped.

### Launch

The SLURM script orchestrates all three phases in one job — vLLM server startup (with health-check waiting up to 90 min for GDN JIT), trainer launch, and cleanup:

```bash
export OPENROUTER_API_KEY="your-api-key"
sbatch grpo/run_qwen3next_fullparam_grpo.sh
```

### Basic Usage

```bash
# 1. Start the vLLM weight-sync server on the inference node
python grpo/vllm_trl_serve.py \
    --model /path/to/Qwen3-Next-80B-A3B-Instruct \
    --language-model-only \
    --tensor-parallel-size 8 \
    --distributed-executor-backend mp \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enforce-eager \
    --max-num-seqs 128

# 2. Launch the trainer on the training nodes
accelerate launch \
    --multi_gpu \
    --num_machines 4 \
    --num_processes 32 \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${TRAINER_MASTER_IP} \
    --main_process_port ${TRAINER_MASTER_PORT} \
    --mixed_precision no \
    grpo/run_qwen3next_fullparam_grpo.py \
    --model /path/to/Qwen3-Next-80B-A3B-Instruct \
    --layers 48 \
    --dataset /path/to/train.jsonl \
    --seq_len 8192 \
    --batch_size 1 \
    --grpo_group_size 4 \
    --steps 200 \
    --lr 1e-5 \
    --gen_max_new_tokens 1024 \
    --use_vllm \
    --vllm_base_url "http://VLLM_IP:8000" \
    --judge_backend openrouter \
    --judge_model_name qwen/qwen3-235b-a22b \
    --judge_concurrency 100 \
    --wandb --wandb_project qwen3next-fullparam-grpo \
    --save_dir ./output/qwen3next-fullparam-grpo
```

> `fsdp_vllm_client.py` must be importable by the trainer (same directory / `PYTHONPATH`). Generation without `--use_vllm` is not supported for this model size.

### Full Argument Reference

#### Model & Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Path to base model |
| `--layers` | int | 48 | Number of decoder layers (Qwen3-Next = 48) |
| `--dataset` | str | **required** | Path to ChatML JSONL training data (optional `metadata.original_truth`) |
| `--seq_len` | int | 4096 | Maximum sequence length for the policy forward pass |
| `--freeze_embeddings` | flag | off | Freeze `embed_tokens` and `lm_head` |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--grpo_group_size` | int | 4 | Number of candidate responses per prompt |
| `--steps` | int | 500 | Maximum optimizer steps |
| `--epochs` | int | 0 | Number of epochs (>0 overrides `--steps` with `epochs × ceil(dataset / global_batch)`) |
| `--batch_size` | int | 1 | Prompts per GPU per step |
| `--lr` | float | 1e-5 | Learning rate (AdamW, betas=(0.9, 0.98)) |
| `--weight_decay` | float | 0.01 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold (0 to disable) |

#### Generation Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--gen_max_new_tokens` | int | 1024 | Maximum tokens for candidate responses |
| `--gen_temperature` | float | 0.7 | Sampling temperature |
| `--gen_top_p` | float | 0.8 | Top-p (nucleus) sampling |
| `--gen_repetition_penalty` | float | 1.05 | Repetition penalty |

#### vLLM Integration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_vllm` | flag | off | Enable vLLM generation + weight sync (required in practice) |
| `--vllm_base_url` | str | None | vLLM server URL (single server; `http://localhost:8000` if omitted) |
| `--vllm_base_urls` | str | None | Comma-separated URLs for multiple servers (takes priority; prompts are sharded round-robin) |
| `--vllm_update_every` | int | 1 | Sync weights to vLLM every N optimizer steps |
| `--vllm_force_n1` | flag | off | Emulate n>1 sampling as repeated n=1 calls (debugging) |
| `--vllm_connection_timeout` | int | 600 | Server health-check timeout at client startup (seconds) |
| `--vllm_max_model_len` | int | 8192 | Must match the vLLM server's `--max-model-len`; used for the prompt token budget |
| `--prompt_len_margin` | int | 16 | Safety margin tokens for chat template / special tokens |

#### Overlong Prompt Filter

Prompts longer than `vllm_max_model_len - gen_max_new_tokens - prompt_len_margin` tokens (after chat templating) are dropped from the dataset by default, preventing vLLM request failures.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--no_filter_overlong_prompts` | flag | filter on | Disable the overlong-prompt filter |
| `--log_overlong_drops` | flag | off | Print drop logs for overlong prompts (rank 0 only) |

#### Judge Settings

The judge receives the prompt, `original_truth`, the reference answer, and the candidate, and must return JSON `{"score": <0-9 integer>, "reason": "..."}`. Score parsing falls back to regex extraction; scores are clamped to [0, 9]. Unrecoverable failures become NaN and are excluded from advantage normalization. HTTP 429/5xx responses are retried with exponential backoff and jitter.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--judge_backend` | str | openrouter | Judge API backend (`openrouter` / `openai`) |
| `--judge_model_name` | str | None | Judge model name (judging is disabled without it) |
| `--judge_api_base` | str | None | API base URL (default: `https://openrouter.ai/api` for openrouter, else `http://localhost:8000`) |
| `--judge_api_key` | str | None | API key (falls back to `OPENROUTER_API_KEY`, then `JUDGE_API_KEY` env vars) |
| `--judge_max_new_tokens` | int | 2048 | Maximum generation tokens for the judge |
| `--judge_temperature` | float | 0.0 | Judge sampling temperature |
| `--judge_max_retries` | int | 1000 | Maximum retry count per judge call |
| `--judge_concurrency` | int | 100 | Parallel judge worker threads (also sizes the HTTP connection pool) |

#### Reward Settings

| Argument | Type | Default | Description |
|---|---|---|---|
| `--reward_scale` | float | 1.0 | Multiplier applied to the normalized advantage |
| `--reward_clip` | float | 1.0 | Advantage clamp range (`±reward_clip`; <=0 disables) |
| `--ema_alpha` | float | 0.05 | EMA smoothing factor for the reward baseline (monitoring only) |
| `--reward_cache` | str | judge_score_cache.jsonl | Judge score cache file (automatically suffixed `.rank{N}`) |

#### Save & Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | ckpts_qwen3next_grpo | Checkpoint save directory (final sharded save) |
| `--log_every` | int | 1 | Log output interval (steps) |
| `--debug_print` | flag | off | Enable debug output |
| `--wandb` | flag | off | Enable W&B logging (metrics + per-step sample tables) |
| `--wandb_project` | str | qwen3next_fullparam_grpo | W&B project name |
| `--wandb_run_name` | str | None | W&B run name |
| `--wandb_entity` | str | None | W&B entity |

---

## 4. vLLM Inference Server

`grpo/vllm_trl_serve.py` is a custom FastAPI server wrapping a vLLM `LLM` engine with TRL's `WeightSyncWorkerExtension`, so FSDP trainer weights can be pushed into a running server over NCCL. On startup it runs a warmup generation to trigger GDN FlashInfer JIT compilation (unless `--skip-warmup`).

### Server Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Model path |
| `--revision` | str | None | Model revision |
| `--tensor-parallel-size` | int | 1 | Tensor parallel size |
| `--pipeline-parallel-size` | int | 1 | Pipeline parallel size |
| `--host` | str | 0.0.0.0 | Bind host |
| `--port` | int | 8000 | Bind port |
| `--gpu-memory-utilization` | float | 0.90 | GPU memory utilization |
| `--dtype` | str | auto | Model dtype (`bfloat16` recommended) |
| `--max-model-len` | int | None | Maximum model context length |
| `--max-num-seqs` | int | 64 | Max concurrent sequences in the scheduler |
| `--enable-prefix-caching` | flag | off | Enable prefix caching (NOT supported by Qwen3-Next — leave off) |
| `--enforce-eager` | flag | off | Disable CUDA graphs (required for Qwen3-Next) |
| `--trust-remote-code` | flag | off | Trust remote model code |
| `--language-model-only` | flag | off | Skip vision encoder (`limit_mm_per_prompt={"image":0,"video":0}`) |
| `--distributed-executor-backend` | str | ray | Executor backend (`ray` / `mp` / `uni`); single node uses `mp` (no Ray needed) |
| `--kv-cache-dtype` | str | auto | KV cache dtype |
| `--log-level` | str | info | uvicorn log level |
| `--skip-warmup` | flag | off | Skip the warmup generation (not recommended for GDN models) |

### HTTP Endpoints

All endpoints accept both `/path` and `/path/` forms.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (`{"status": "ok"}`) |
| `/get_world_size` | GET | Returns `{"world_size": tensor_parallel_size}` |
| `/get_tensor_parallel_size` | GET | Returns `{"tensor_parallel_size": N}` |
| `/init_communicator` | POST | Picks an open port, fires `collective_rpc("init_communicator")` in the background, returns `{host, port, world_size}` (world_size = TP + 1 for the trainer rank) |
| `/update_named_param` | POST | Receives `{name, dtype, shape}` metadata, fires the worker-side NCCL broadcast receive in the background |
| `/generate` | POST | Batch generation; body `{prompts, n, max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, guided_decoding_regex}`; returns `{"completion_ids": [[token ids], ...]}` |
| `/reset_prefix_cache` | POST | Invalidate stale KV prefix cache (called by the client after every weight update) |
| `/close_communicator` | POST | Tear down the NCCL communicator on all workers |
| `/force_exit` | POST | SIGKILL the server process after 0.5 s — triggers the shell auto-restart loop (used by client recovery) |

Weight-sync RPCs run through a single-threaded executor to serialize updates and avoid races.

### FSDP vLLM Client (`grpo/fsdp_vllm_client.py`)

`FSDPVLLMClient` (aliased `VLLMClient`) combines HTTP generation with NCCL weight transfer. Constructor: `FSDPVLLMClient(base_url="http://localhost:8000", connection_timeout=240, max_retries=10)` — waits for `/health` before returning.

| Method | Description |
|---|---|
| `init_communicator(device)` | Rank 0 only: POST `/init_communicator`, join the `StatelessProcessGroup` TCP rendezvous, create the `PyNcclCommunicator` (trainer = last rank) |
| `update_named_param(name, tensor)` | Rank 0 only: POST param metadata, then NCCL-broadcast the tensor to all vLLM workers (blocks until received) |
| `update_model_params(model, trainable_only)` | Send all params of a non-FSDP model, then reset the prefix cache |
| `update_model_params_streaming_fsdp(model, trainable_only)` | Stream FSDP-sharded weights one FSDP unit at a time via `summon_full_params(writeback=False, rank0_only=True)`; all ranks must call it (collective), only rank 0 broadcasts; resets the prefix cache afterwards |
| `generate(prompts, n, max_tokens, temperature, top_p, repetition_penalty, min_p, ...)` | POST `/generate`; returns `{"completion_ids": ...}`. First call uses a 7200 s timeout with retries (JIT safety net), later calls 600 s. On failure after the first success, runs connection recovery and raises `VLLMRecoveryNeeded` |
| `reset_prefix_cache()` | POST `/reset_prefix_cache` (mandatory after weight updates — stale KV blocks otherwise cause hangs / wrong outputs) |
| `close_communicator()` | Notify the server and drop the local NCCL communicator |

`VLLMRecoveryNeeded` is raised after the client has re-established the connection to a restarted server; the caller must resync weights (collective across all ranks) and retry generation. The internal recovery sequence: `/force_exit` → close stale NCCL comm + reset the HTTP session → wait for auto-restart (up to 1800 s) → verify with a 1-token generate (5 attempts) → re-init the NCCL communicator.

### Triton Worker Extension (`grpo/triton_worker_extension.py`)

`TritonAllocatorWeightSyncWorkerExtension` subclasses TRL's `WeightSyncWorkerExtension` and installs a Triton runtime memory allocator (at import time and again at instantiation). Triton 3.x kernels that perform runtime allocations (MoE routing / GDN / TMA scratch) otherwise fail with `Kernel requires a runtime memory allocation, but no allocator was set`. The module is loaded into every vLLM worker via `worker_extension_cls="triton_worker_extension.TritonAllocatorWeightSyncWorkerExtension"`, so it must sit in the same directory as `vllm_trl_serve.py` (the server adds that directory to `PYTHONPATH`, and ships it as the Ray `working_dir` when the Ray backend is used).

### Weight Sync Protocol

1. `init_communicator` — server picks an open port and fires a background `collective_rpc`; the client joins the `StatelessProcessGroup` rendezvous at `{host, port}` as the last rank (world_size = TP + 1) and both sides create `PyNcclCommunicator`
2. `update_named_param` — client POSTs `{name, dtype, shape}` (server fires the receive in the background), then calls `pynccl.broadcast(src=trainer_rank)`; NCCL synchronization guarantees the workers received the tensor before the client proceeds to the next parameter
3. `reset_prefix_cache` — invalidate stale KV blocks after all parameters are sent
4. `close_communicator` — fire-and-forget cleanup

---

## 5. Model Merging

`merge/merge_qwen80b_full_model.py` merges FSDP sharded checkpoints (`model_sharded.rank*.pt`) into a vLLM-compatible model. The key problem it solves: the training-time model stores MoE experts in **stacked** form (`...mlp.experts.down_proj` → `[E, out, in]`, `...mlp.experts.gate_up_proj` → `[E, 2*intermediate, in]`), while the base-model safetensors / vLLM expect **per-expert** keys (`...mlp.experts.E.{gate_proj,up_proj,down_proj}.weight`). Instead of `save_pretrained`, it uses the base model's safetensors shards as a template and replaces weights shard by shard, preserving the exact file layout.

- **Runs on**: 1 node, CPU only (no GPU required; the SLURM script reserves 240 CPUs and unlimited memory)
- **Reads**: checkpoints saved by any of the three trainers (patterns `model_sharded.rank{r}.pt`, `params.rank{r}.pt`, `attn_params.rank{r}.pt`; world size auto-detected from `model_sharded_meta.json` or glob)

### Processing Flow

1. **Phase 1 — Streaming merge**: two-pass over rank files (metadata scan → allocate full tensors → streaming write); FSDP `ShardedTensor` objects are read via an unpickle shim without initializing a process group; strict mode verifies dim-0 contiguity and completeness of every parameter
2. **Phase 2 — Template-based save**: load each base-model safetensors shard, replace matching keys from the merged params (converting stacked experts to individual `gate_proj`/`up_proj`/`down_proj` slices), save under the same shard name, copy `model.safetensors.index.json` unchanged
3. **Phase 3 — Copy config / tokenizer / custom code**: `config.json`, `tokenizer*.json`, `special_tokens_map.json`, `generation_config.json`, `chat_template.jinja`, and any `*.py` from the base model directory

### Launch

```bash
sbatch merge/merge_qwen80b_full_model.sh
```

### Basic Usage

```bash
python merge/merge_qwen80b_full_model.py \
    --base_model /path/to/Qwen3-Next-80B-A3B-Instruct \
    --checkpoint_dir /path/to/checkpoint \
    --out_dir /path/to/merged_model \
    --num_layers 48 \
    --world_size 40

# Verify with vLLM
vllm serve /path/to/merged_model --trust-remote-code
```

### Full Argument Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--base_model` | str | **required** | HF local path to the base model (must contain `model.safetensors.index.json`) |
| `--checkpoint_dir` | str | **required** | FSDP sharded checkpoint directory |
| `--out_dir` | str | **required** | Output directory |
| `--num_layers` | int | 48 | Number of model layers |
| `--world_size` | int | 0 | Number of rank files (0 = auto-detect from meta JSON / glob) |
| `--no_strict` | flag | off | Skip the dim-0 contiguity / completeness verification |

---

## 6. SLURM Configuration

### Environment Variables

Settings common to all training jobs (see the `.sh` scripts):

```bash
# Modules
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cuda/12.4

# Network interface (adjust to your cluster)
export TARGET_IFNAME="eth0"

# NCCL
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL                     # Use NVLink for intra-node P2P
export NCCL_IB_DISABLE=0                      # Enable InfiniBand
export NCCL_SOCKET_IFNAME="${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME="${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="${TARGET_IFNAME}"

# Timeouts (7200 s — covers GDN FlashInfer JIT compilation on first run)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

# PyTorch
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1                  # torch.compile not supported
export OMP_NUM_THREADS=2
ulimit -n 262144
```

GRPO additionally sets:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn     # mp executor backend
export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer"   # persist GDN JIT cache
```

### Node Configuration

| Pipeline | Nodes | GPUs/Node | Total GPUs | Notes |
|---|---|---|---|---|
| SFT | 5 | 8 | 40 | accelerate launch, FSDP FULL_SHARD |
| DPO | 5 | 8 | 40 | accelerate launch, FSDP FULL_SHARD |
| GRPO (trainer) | 4 | 8 | 32 | accelerate launch, FSDP FULL_SHARD |
| GRPO (vLLM) | 1 | 8 | 8 | tensor_parallel_size=8, mp backend, same SLURM job |
| Merge | 1 | 0 (CPU) | — | CPU-only, 240 CPUs, large RAM |

### Accelerate Launch Template

```bash
accelerate launch \
    --multi_gpu \
    --num_machines ${NNODES} \
    --num_processes $((NNODES * 8)) \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --mixed_precision no \
    script.py [args...]
```

The GRPO job script additionally runs the vLLM server on node 0 via SSH inside an auto-restart loop with a health watchdog (probes `/health` every 10 s; after 36 consecutive failures following a healthy state, the server is SIGKILLed and restarted; `/dev/shm/vllm_*` is cleaned between restarts).

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Training data (ChatML JSONL / DPO prompt-chosen-rejected)│
└────────────┬─────────────────────────────────────────────┘
             ▼
┌──────────────────────────────────────────────────────────┐
│  FSDP FULL_SHARD  (Qwen3-Next-80B-A3B, BF16)             │
│  rank 0: CPU load / other ranks: meta device             │
│  → sync_module_states broadcast                          │
├──────────────┬──────────────┬────────────────────────────┤
│  1. SFT      │  2. DPO      │  3. GRPO                   │
│  (packing,   │  (concat fwd,│  (vLLM generate →          │
│   assistant  │   ref logp   │   LLM judge 0–9 →          │
│   masking)   │   precompute)│   group-norm advantage)    │
└──────────────┴──────────────┴──────────┬─────────────────┘
                                         │ NCCL weight broadcast
                                         ▼
                          ┌────────────────────────────┐
                          │  vLLM Server (TP=8)        │
                          │  WeightSyncWorkerExtension │
                          └────────────────────────────┘
             ▼
┌──────────────────────────────────────────────────────────┐
│  Sharded checkpoint (model_sharded.rank*.pt + meta JSON) │
└────────────┬─────────────────────────────────────────────┘
             ▼
┌──────────────────────────────────────────────────────────┐
│  5. Merge (stacked → per-expert, template-based save)    │
│  → vLLM-compatible safetensors → vllm serve              │
└──────────────────────────────────────────────────────────┘
```

## Memory Estimates (BF16, FSDP FULL_SHARD)

| Item | 40 GPUs (SFT/DPO) | 32 GPUs (GRPO) |
|---|---|---|
| Weights | 4 GB/GPU | 5 GB/GPU |
| Adam states | 16 GB/GPU | 20 GB/GPU |
| Gradients | 4 GB/GPU | 5 GB/GPU |
| **Total** | **~24 GB/GPU** | **~30 GB/GPU** |

Fits comfortably on H100 80GB (activations are kept small by gradient checkpointing; DPO's concat forward adds ~4–6 GB).

## System Requirements

- CUDA 12.4 driver
- conda (Python 3.11)
- SLURM job scheduler
- Multi-node H100 cluster (5+ nodes × 8 GPUs)
- High-speed interconnect (InfiniBand / RoCE) for NCCL

## Notes

- **FSDP FULL_SHARD** + `sync_module_states=True` (rank 0 loads BF16 weights on CPU; other ranks build the model on the meta device with a `_load_pretrained_model` no-op patch, so CPU RAM is only consumed once per node)
- **Accelerate** + `GradientAccumulationPlugin(sync_each_batch=True)` for SFT/DPO (FSDP grad-buffer OOM mitigation)
- `transformer_auto_wrap_policy` wraps at `Qwen3NextDecoderLayer` granularity (class discovered dynamically from `transformers.models`, with a fallback to inspecting the instantiated model)
- NCCL / distributed timeouts are set to 7200 s to survive GDN FlashInfer kernel JIT compilation on first run
- Checkpoints are always saved as FSDP `SHARDED_STATE_DICT` per-rank files; run the merge tool before serving with vLLM

## License

[MIT](LICENSE)

## Acknowledgements

This program is based on results obtained from a project, JPNP25006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
