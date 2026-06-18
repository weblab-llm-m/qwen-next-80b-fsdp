#!/bin/bash
#SBATCH --job-name=qwen3next-fullparam-sft
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX,node-XX,node-XX,node-XX,node-XX
#SBATCH --nodes=5
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
#SBATCH --exclusive

set -euo pipefail

echo "=== Qwen3-Next-80B-A3B Full-Parameter SFT (FSDP, BF16) ==="
echo "開始時刻: $(date)"

mkdir -p outputs/logs

# ============================================================
# ネットワーク設定
# ============================================================
export TARGET_IFNAME="eth0"

module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cuda/12.4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0

export GLOO_SOCKET_IFNAME="${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="${TARGET_IFNAME}"

export OMP_NUM_THREADS=2
ulimit -n 262144 || true

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

# ============================================================
# conda 環境 (GRPO と同じ統一環境)
# ============================================================
CONDA_ENV="qwen3next"

eval "$(conda shell.bash hook)"
set +u
conda activate "${CONDA_ENV}"
set -u

# ============================================================
# マスターノード IP 取得
# ============================================================
MASTER_HOSTNAME="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"

HOST_ADDR=$(srun --nodes=1 --ntasks=1 -w "${MASTER_HOSTNAME}" \
    ip -4 addr show "${TARGET_IFNAME}" \
    | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)

if [ -z "${HOST_ADDR}" ]; then
    echo "WARNING: ${TARGET_IFNAME} のIP取得に失敗。ホスト名を使用します。"
    HOST_ADDR="${MASTER_HOSTNAME}"
fi

export MASTER_ADDR="${HOST_ADDR}"
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR  : ${MASTER_ADDR}"
echo "MASTER_PORT  : ${MASTER_PORT}"
echo "SLURM_NNODES : ${SLURM_NNODES}"
echo "Total GPUs   : $((SLURM_NNODES * 8))"

# ============================================================
# 設定値
# ============================================================
# >>> モデルパスを環境に合わせて変更 <<<
BASE_MODEL="/path/to/base_model"

# >>> データセットパスを環境に合わせて変更 <<<
DATASET="/path/to/train_sft.jsonl"

OUTPUT_DIR="./output/qwen3next-fullparam-sft"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch はスクリプトをスプールへコピーして実行するため、BASH_SOURCE が
# リポジトリ外を指す場合は提出ディレクトリ基準で探索(リポジトリルート or sft/ から提出)
if [[ ! -f "${SCRIPT_DIR}/run_qwen3next_fullparam_sft.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  for _cand in "${SLURM_SUBMIT_DIR}/sft" "${SLURM_SUBMIT_DIR}"; do
    if [[ -f "${_cand}/run_qwen3next_fullparam_sft.py" ]]; then SCRIPT_DIR="${_cand}"; break; fi
  done
fi
if [[ ! -f "${SCRIPT_DIR}/run_qwen3next_fullparam_sft.py" ]]; then
  echo "ERROR: run_qwen3next_fullparam_sft.py が見つかりません (SCRIPT_DIR=${SCRIPT_DIR})。リポジトリルートから sbatch してください" >&2
  exit 1
fi

SCRIPT="${SCRIPT_DIR}/run_qwen3next_fullparam_sft.py"

# ────────────────────────────────────────────────────────
# メモリ見積もり (5 nodes x 8 GPU = 40 GPU, FSDP FULL_SHARD):
#   Weights  : 160GB / 40 = 4.0 GB/GPU
#   Adam     : 640GB / 40 = 16.0 GB/GPU
#   Grads    : 160GB / 40 = 4.0 GB/GPU
#   Total    : ~24 GB/GPU + activations (grad checkpointing)
#   H100 80GB -> 十分余裕あり
# ────────────────────────────────────────────────────────
# バッチ設定:
#   micro_batch=1 x grad_accum=4 x 40 GPUs = effective batch 160
BATCH_SIZE=1
GRAD_ACCUM=4
SEQ_LEN=2048
LR="1e-5"
EPOCHS=1

# W&B 設定 (空文字で無効化)
WANDB_PROJECT="qwen3next-sft"
WANDB_RUN_NAME="fullparam-packing"

echo "=== accelerate launch ==="
echo "micro_batch=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM}, GPUs=$((SLURM_NNODES * 8)), packing=on"
echo "effective_batch=$((BATCH_SIZE * GRAD_ACCUM * SLURM_NNODES * 8))"
echo "seq_len=${SEQ_LEN}, lr=${LR}, epochs=${EPOCHS}"

# ============================================================
# 実行
# ============================================================
srun -l --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR='"${MASTER_ADDR}"'
export MASTER_PORT='"${MASTER_PORT}"'

NODE_RANK="${SLURM_NODEID}"
NNODES='"${SLURM_NNODES}"'
NPROC=8

echo "[Node ${NODE_RANK}] hostname=$(hostname), MASTER=${MASTER_ADDR}:${MASTER_PORT}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

accelerate launch \
  --multi_gpu \
  --num_machines ${NNODES} \
  --num_processes $((NNODES * NPROC)) \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision no \
  '"${SCRIPT}"' \
    --model '"${BASE_MODEL}"' \
    --num_layers 48 \
    --dataset '"${DATASET}"' \
    --seq_len '"${SEQ_LEN}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --gradient_accumulation_steps '"${GRAD_ACCUM}"' \
    --packing \
    --epochs '"${EPOCHS}"' \
    --steps 0 \
    --lr '"${LR}"' \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --wandb_project '"${WANDB_PROJECT}"' \
    --wandb_run_name '"${WANDB_RUN_NAME}"' \
    --save_dir '"${OUTPUT_DIR}"'
'

echo "=== ジョブ完了: $(date) ==="
