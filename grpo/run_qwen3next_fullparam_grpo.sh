#!/bin/bash
#SBATCH --job-name=qwen3next-fullparam-grpo
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=5
#SBATCH --nodelist=node-XX,node-XX,node-XX,node-XX,node-XX
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
#SBATCH --exclusive

# ============================================================
# Qwen3-Next-80B-A3B Full-Parameter GRPO
#
# Node layout (5 nodes, 40 GPUs):
#   Node 0       : vLLM server (TP=8, mp backend, 8 GPU)
#   Nodes 1-4    : Trainer     (FSDP FULL_SHARD, 32 GPU)
#
# Memory budget (Trainer, 32 GPU):
#   Weights  : 160GB / 32 = 5GB/GPU
#   Adam     : 640GB / 32 = 20GB/GPU
#   Grads    : 160GB / 32 = 5GB/GPU
#   Total    : ~30GB/GPU (H100 80GB → OK)
#
# vLLM: single node → mp backend (no Ray needed)
# NOTE: Qwen3-Next does NOT support prefix caching
# ============================================================

set -uo pipefail

echo "================================================================"
echo "  Qwen3-Next-80B-A3B Full-Param GRPO"
echo "  vLLM 1 node (TP=8) + Trainer 4 nodes (32 GPU)"
echo "  Start: $(date)"
echo "================================================================"

mkdir -p outputs/logs

# ============================================================
# Network
# ============================================================
export TARGET_IFNAME="eth0"

module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cuda/12.4

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export GLOO_SOCKET_IFNAME="${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="${TARGET_IFNAME}"

export OMP_NUM_THREADS=2
ulimit -n 262144 || true

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

# FlashInfer GDN kernel JIT cache
export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer"
mkdir -p "${FLASHINFER_CACHE_DIR}"

# VLLM_WORKER_MULTIPROC_METHOD for mp backend
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ============================================================
# Conda environment (統合: vLLM + Trainer 同一環境)
# ============================================================
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
CONDA_ENV="${CONDA_BASE}/envs/qwen3next"

echo "Conda env: ${CONDA_ENV}"

# ============================================================
# Node allocation
# ============================================================
ALL_NODES=($(scontrol show hostnames "${SLURM_NODELIST}"))
VLLM_NODE_COUNT=1
TRAINER_NODE_COUNT=4

VLLM_NODE="${ALL_NODES[0]}"
TRAINER_NODES=("${ALL_NODES[@]:${VLLM_NODE_COUNT}}")
TRAINER_NNODES=${#TRAINER_NODES[@]}

echo "vLLM node    : ${VLLM_NODE}"
echo "Trainer nodes: ${TRAINER_NODES[*]} (${TRAINER_NNODES} nodes, $((TRAINER_NNODES * 8)) GPUs)"

# ============================================================
# IP addresses
# ============================================================
get_node_ip() {
    ssh -o StrictHostKeyChecking=no "$1" \
        "ip -4 addr show ${TARGET_IFNAME} 2>/dev/null \
         | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1"
}

VLLM_IP=$(get_node_ip "${VLLM_NODE}")
TRAINER_MASTER_IP=$(get_node_ip "${TRAINER_NODES[0]}")

echo "vLLM IP         : ${VLLM_IP}"
echo "Trainer master IP: ${TRAINER_MASTER_IP}"

if [ -z "${VLLM_IP}" ] || [ -z "${TRAINER_MASTER_IP}" ]; then
    echo "ERROR: Failed to get IP addresses"
    exit 1
fi

# ============================================================
# Config
# ============================================================
BASE_MODEL="/path/to/base_model"
DATASET="/path/to/train_sft.jsonl"
OUTPUT_DIR="./output/qwen3next-fullparam-grpo-$(date +%Y%m%d-%H%M)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch はスクリプトをスプールへコピーして実行するため、BASH_SOURCE が
# リポジトリ外を指す場合は提出ディレクトリ基準で探索(リポジトリルート or grpo/ から提出)
if [[ ! -f "${SCRIPT_DIR}/run_qwen3next_fullparam_grpo.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  for _cand in "${SLURM_SUBMIT_DIR}/grpo" "${SLURM_SUBMIT_DIR}"; do
    if [[ -f "${_cand}/run_qwen3next_fullparam_grpo.py" ]]; then SCRIPT_DIR="${_cand}"; break; fi
  done
fi
if [[ ! -f "${SCRIPT_DIR}/run_qwen3next_fullparam_grpo.py" ]]; then
  echo "ERROR: run_qwen3next_fullparam_grpo.py が見つかりません (SCRIPT_DIR=${SCRIPT_DIR})。リポジトリルートから sbatch してください" >&2
  exit 1
fi

SCRIPT="${SCRIPT_DIR}/run_qwen3next_fullparam_grpo.py"

# vLLM
VLLM_PORT=8000
VLLM_TP=8
VLLM_GPU_UTIL=0.90
VLLM_MAX_NUM_SEQS=128

# GRPO
GRPO_GROUP_SIZE=4
BATCH_SIZE=1
GEN_MAX_TOKENS=1024
GEN_TEMPERATURE=0.7

# Judge
JUDGE_BACKEND="openrouter"
JUDGE_MODEL="qwen/qwen3-235b-a22b"
JUDGE_API_BASE="https://openrouter.ai/api"
JUDGE_CONCURRENCY=100

# W&B
WANDB_PROJECT="qwen3next-fullparam-grpo"
WANDB_RUN_NAME="fullparam-5node-$(date +%Y%m%d-%H%M)"

# ============================================================
# Environment setup script (統合: 1つだけ)
# ============================================================

ENV_SETUP="${SCRIPT_DIR}/.env_setup_qwen3next_${SLURM_JOB_ID}.sh"
cat > "${ENV_SETUP}" << ENVEOF
#!/bin/bash
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

for init in /etc/profile.d/modules.sh /usr/share/modules/init/bash; do
    [ -f "\$init" ] && source "\$init" 2>/dev/null && break
done
module load nccl/2.22.3 2>/dev/null || true
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true

export TARGET_IFNAME="${TARGET_IFNAME}"
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="\${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME="\${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="\${TARGET_IFNAME}"
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${SCRIPT_DIR}:\${PYTHONPATH:-}"

export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer"
mkdir -p "\${FLASHINFER_CACHE_DIR}" 2>/dev/null || true
ENVEOF
chmod +x "${ENV_SETUP}"

echo "Env script: ${ENV_SETUP}"

# ============================================================
# TRL compatibility patches (vLLM env)
# ============================================================
source "${ENV_SETUP}"

echo "[PATCH] vllm_ascend stub..."
python3 << 'PATCH_VLLM_ASCEND'
import os, site
try:
    sp = site.getsitepackages()[0]
except Exception:
    import sys
    sp = os.path.join(os.path.dirname(sys.executable), "..", "lib",
                      f"python{sys.version_info.major}.{sys.version_info.minor}",
                      "site-packages")
pkg = os.path.join(sp, "vllm_ascend", "distributed", "device_communicators")
os.makedirs(pkg, exist_ok=True)
# __init__.py は空（パッケージとしてimport可能にする）
for d in [os.path.join(sp, "vllm_ascend"),
          os.path.join(sp, "vllm_ascend", "distributed"),
          pkg]:
    init = os.path.join(d, "__init__.py")
    with open(init, "w") as f:
        f.write("")
# pyhccl.py で本物のPyNcclCommunicatorを再エクスポート
# TRLが PyHcclCommunicator as PyNcclCommunicator でimportしても
# 結局本物のNCCL版が使われる
pyhccl = os.path.join(pkg, "pyhccl.py")
with open(pyhccl, "w") as f:
    f.write("from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator as PyHcclCommunicator\n")
print("[PATCH] vllm_ascend stub: pyhccl.py re-exports PyNcclCommunicator")
PATCH_VLLM_ASCEND

python3 -c "
from trl.scripts.vllm_serve import WeightSyncWorkerExtension
print('[PATCH] ✅ WeightSyncWorkerExtension OK')
" 2>&1 || echo "[PATCH] ⚠️  WeightSyncWorkerExtension import failed"

# ============================================================
# Phase 1: vLLM server (single node, mp backend — no Ray)
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 1: vLLM server (mp backend, single node)"
echo "  TP=${VLLM_TP}, model=$(basename ${BASE_MODEL})"
echo "  max_num_seqs=${VLLM_MAX_NUM_SEQS}"
echo "  NOTE: Qwen3-Next does NOT support prefix caching"
echo "  NOTE: First run may take 40-60+ min for GDN JIT"
echo "================================================================"

VLLM_LOG="outputs/logs/vllm-${SLURM_JOB_ID}.log"

ssh -o StrictHostKeyChecking=no "${VLLM_NODE}" "bash -c '
    source ${ENV_SETUP}
    export VLLM_HOST_IP=${VLLM_IP}
    echo \"[vLLM] Launching on \$(hostname) (${VLLM_IP})\"
    echo \"[vLLM] Python: \$(python --version)\"
    echo \"[vLLM] vLLM: \$(python -c \"import vllm; print(vllm.__version__)\" 2>/dev/null)\"
    echo \"[vLLM] TP=${VLLM_TP}, port=${VLLM_PORT}, backend=mp\"
    echo \"[vLLM] FLASHINFER_CACHE_DIR=\${FLASHINFER_CACHE_DIR}\"

    RESTART_COUNT=0

    while true; do
        if [ \${RESTART_COUNT} -gt 0 ]; then
            echo \"[vLLM] ========== RESTART #\${RESTART_COUNT} at \$(date) ==========\"
            # mp backend: just kill any leftover processes
            pkill -9 -f vllm_trl_serve 2>/dev/null || true
            pkill -9 -f \"vllm\" 2>/dev/null || true
            sleep 5
            # Clean shared memory
            rm -rf /dev/shm/vllm_* 2>/dev/null || true
        fi

        echo \"[vLLM] Starting server at \$(date)\"
        python3 ${SCRIPT_DIR}/vllm_trl_serve.py \
            --model ${BASE_MODEL} \
            --language-model-only \
            --tensor-parallel-size ${VLLM_TP} \
            --distributed-executor-backend mp \
            --port ${VLLM_PORT} \
            --dtype bfloat16 \
            --trust-remote-code \
            --gpu-memory-utilization ${VLLM_GPU_UTIL} \
            --max-model-len 8192 \
            --enforce-eager \
            --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
            2>&1 &
        SERVER_PID=\$!
        echo \"[vLLM] PID=\${SERVER_PID}\"

        # Health watchdog
        WAS_HEALTHY=0
        HEALTH_FAIL_COUNT=0
        HEALTH_FAIL_LIMIT=36  # 6 min

        while kill -0 \${SERVER_PID} 2>/dev/null; do
            if curl -s --max-time 5 http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
                WAS_HEALTHY=1
                HEALTH_FAIL_COUNT=0
            else
                if [ \${WAS_HEALTHY} -eq 1 ]; then
                    HEALTH_FAIL_COUNT=\$((HEALTH_FAIL_COUNT + 1))
                    if [ \${HEALTH_FAIL_COUNT} -ge \${HEALTH_FAIL_LIMIT} ]; then
                        echo \"[vLLM] Watchdog: unresponsive — SIGKILL\"
                        kill -9 \${SERVER_PID} 2>/dev/null || true
                        break
                    fi
                fi
            fi
            sleep 10
        done

        wait \${SERVER_PID} 2>/dev/null
        EXIT_CODE=\$?
        RESTART_COUNT=\$((RESTART_COUNT + 1))
        echo \"[vLLM] Exited code=\${EXIT_CODE}, restart #\${RESTART_COUNT} in 10s...\"
        sleep 10
    done
'" > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "[vLLM] Background PID=${VLLM_PID}, log=${VLLM_LOG}"

# Wait for vLLM health (up to 90 min for GDN JIT)
echo "[vLLM] Waiting for http://${VLLM_IP}:${VLLM_PORT}/health ..."
echo "[vLLM] First run may take 40-60+ min for GDN JIT compilation"
VLLM_READY=0
WAIT_START=$(date +%s)
for i in $(seq 1 540); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        "http://${VLLM_IP}:${VLLM_PORT}/health" 2>/dev/null || echo "000")

    if [ "${HTTP_CODE}" = "200" ]; then
        WAIT_END=$(date +%s)
        WAIT_SECS=$((WAIT_END - WAIT_START))
        echo "[vLLM] ✅ Ready! (waited ${WAIT_SECS}s = $((WAIT_SECS/60))m)"
        VLLM_READY=1
        break
    fi

    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died"
        tail -50 "${VLLM_LOG}" 2>/dev/null
        exit 1
    fi

    if [ $((i % 6)) -eq 0 ]; then
        ELAPSED=$(($(date +%s) - WAIT_START))
        echo "[vLLM] Loading... (${ELAPSED}s = $((ELAPSED/60))m)"
        tail -2 "${VLLM_LOG}" 2>/dev/null | sed 's/^/  /' || true
    fi
    sleep 10
done

if [ "${VLLM_READY}" -ne 1 ]; then
    echo "ERROR: vLLM did not start in time"
    tail -100 "${VLLM_LOG}" 2>/dev/null
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

curl -s "http://${VLLM_IP}:${VLLM_PORT}/get_world_size" 2>/dev/null \
    | python3 -m json.tool 2>/dev/null || true

# ============================================================
# Phase 2: Trainer (4 nodes, 32 GPU)
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 2: Trainer (${TRAINER_NNODES} nodes × 8 GPUs = $((TRAINER_NNODES * 8)) procs)"
echo "================================================================"

TRAINER_MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
TRAINER_NPROC=$((TRAINER_NNODES * 8))
VLLM_URL="http://${VLLM_IP}:${VLLM_PORT}"

echo "Trainer master: ${TRAINER_MASTER_IP}:${TRAINER_MASTER_PORT}"
echo "Trainer procs : ${TRAINER_NPROC}"
echo "vLLM URL      : ${VLLM_URL}"

TRAINER_NODELIST_CSV=$(IFS=,; echo "${TRAINER_NODES[*]}")
TRAINER_NODES_STR="${TRAINER_NODES[*]}"

srun --nodes="${TRAINER_NNODES}" \
     --ntasks="${TRAINER_NNODES}" \
     --ntasks-per-node=1 \
     -w "${TRAINER_NODELIST_CSV}" \
     bash -c '
set -euo pipefail
source '"${ENV_SETUP}"'

echo "[Trainer] Python: $(python --version)"
echo "[Trainer] transformers: $(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)"

# Node rank
TRAINER_NODES_ARR=('"${TRAINER_NODES_STR}"')
MY_HOST=$(hostname -s)
NODE_RANK=-1
for i in "${!TRAINER_NODES_ARR[@]}"; do
    TGT=$(echo "${TRAINER_NODES_ARR[$i]}" | cut -d. -f1)
    if [ "${TGT}" = "${MY_HOST}" ]; then
        NODE_RANK=$i
        break
    fi
done

if [ ${NODE_RANK} -lt 0 ]; then
    echo "ERROR: ${MY_HOST} not in trainer nodes"
    exit 1
fi

NNODES='"${TRAINER_NNODES}"'
NPROC=8

echo "[Trainer] node_rank=${NODE_RANK}/${NNODES} host=${MY_HOST}"

accelerate launch \
  --multi_gpu \
  --num_machines ${NNODES} \
  --num_processes $((NNODES * NPROC)) \
  --machine_rank ${NODE_RANK} \
  --main_process_ip '"${TRAINER_MASTER_IP}"' \
  --main_process_port '"${TRAINER_MASTER_PORT}"' \
  --mixed_precision no \
  '"${SCRIPT}"' \
    --model '"${BASE_MODEL}"' \
    --layers 48 \
    --dataset '"${DATASET}"' \
    --seq_len 8192 \
    --batch_size '"${BATCH_SIZE}"' \
    --grpo_group_size '"${GRPO_GROUP_SIZE}"' \
    --steps 200 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --gen_max_new_tokens '"${GEN_MAX_TOKENS}"' \
    --gen_temperature '"${GEN_TEMPERATURE}"' \
    --gen_top_p 0.8 \
    --gen_repetition_penalty 1.05 \
    --use_vllm \
    --vllm_base_url '"${VLLM_URL}"' \
    --vllm_update_every 1 \
    --vllm_connection_timeout 600 \
    --judge_backend '"${JUDGE_BACKEND}"' \
    --judge_model_name '"${JUDGE_MODEL}"' \
    --judge_api_base '"${JUDGE_API_BASE}"' \
    --judge_max_new_tokens 1024 \
    --judge_concurrency '"${JUDGE_CONCURRENCY}"' \
    --reward_scale 1.0 \
    --reward_clip 1.0 \
    --wandb \
    --wandb_project '"${WANDB_PROJECT}"' \
    --wandb_run_name '"${WANDB_RUN_NAME}"' \
    --log_every 1 \
    --save_dir '"${OUTPUT_DIR}"'
'

TRAINER_EXIT=$?
echo "[Trainer] exit_code=${TRAINER_EXIT}"

# ============================================================
# Phase 3: Cleanup
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 3: Cleanup"
echo "================================================================"

kill ${VLLM_PID} 2>/dev/null || true
ssh -o StrictHostKeyChecking=no "${VLLM_NODE}" \
    "pkill -f vllm_trl_serve 2>/dev/null; true" &
sleep 3

rm -f "${ENV_SETUP}"

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Trainer exit: ${TRAINER_EXIT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  vLLM log: ${VLLM_LOG}"
echo "================================================================"

exit ${TRAINER_EXIT}
