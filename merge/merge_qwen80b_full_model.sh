#!/bin/bash
#SBATCH --job-name=dpo-full-merge
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/dpo_full_merge-%j.out
#SBATCH --error=outputs/logs/dpo_full_merge-%j.err

set -euo pipefail

echo "=== DPO フルモデルマージ v2 (template-based) ==="
echo "開始時刻: $(date)"
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16

ulimit -m unlimited  || true
ulimit -v unlimited  || true
ulimit -d unlimited  || true
ulimit -s unlimited  || true
ulimit -n 262144     || true

mkdir -p outputs/logs

# ============================================================
# Python 環境
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch はスクリプトをスプールへコピーして実行するため、BASH_SOURCE が
# リポジトリ外を指す場合は提出ディレクトリ基準で探索(リポジトリルート or merge/ から提出)
if [[ ! -f "${SCRIPT_DIR}/merge_qwen80b_full_model.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  for _cand in "${SLURM_SUBMIT_DIR}/merge" "${SLURM_SUBMIT_DIR}"; do
    if [[ -f "${_cand}/merge_qwen80b_full_model.py" ]]; then SCRIPT_DIR="${_cand}"; break; fi
  done
fi
if [[ ! -f "${SCRIPT_DIR}/merge_qwen80b_full_model.py" ]]; then
  echo "ERROR: merge_qwen80b_full_model.py が見つかりません (SCRIPT_DIR=${SCRIPT_DIR})。リポジトリルートから sbatch してください" >&2
  exit 1
fi

PYTHON="$(conda info --base)/envs/qwen3next/bin/python"

echo "Python: ${PYTHON} ($(${PYTHON} --version))"
${PYTHON} -c "import torch; import safetensors; print('env check OK: torch=' + torch.__version__)"

# ============================================================
# パス設定
# ============================================================
BASE_MODEL="/path/to/base_model"
CKPT_DIR="/path/to/checkpoint"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/path/to/merged_model_dpo_${TIMESTAMP}"

NUM_LAYERS=48
WORLD_SIZE=40

echo ""
echo "BASE_MODEL   : ${BASE_MODEL}"
echo "CKPT_DIR     : ${CKPT_DIR}"
echo "OUT_DIR      : ${OUT_DIR}"
echo "NUM_LAYERS   : ${NUM_LAYERS}"
echo "WORLD_SIZE   : ${WORLD_SIZE}"

# ============================================================
# 事前チェック
# ============================================================
echo ""

FOUND_FILES=$(find "${CKPT_DIR}" -maxdepth 1 -name "model_sharded.rank*.pt" 2>/dev/null | wc -l || echo "0")
echo "rank ファイル数: ${FOUND_FILES}"

if [ "${FOUND_FILES}" -eq 0 ]; then
    echo "[ERROR] rank ファイルが見つかりません: ${CKPT_DIR}/"
    exit 1
fi

# ベースモデルの index.json 確認
if [ ! -f "${BASE_MODEL}/model.safetensors.index.json" ]; then
    echo "[ERROR] ベースモデルに model.safetensors.index.json がありません"
    exit 1
fi

TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
echo "ノード RAM: ${TOTAL_MEM_GB} GB"
echo ""

# ============================================================
# 実行
# ============================================================
${PYTHON} "${SCRIPT_DIR}/merge_qwen80b_full_model.py" \
    --base_model "${BASE_MODEL}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --out_dir "${OUT_DIR}" \
    --num_layers "${NUM_LAYERS}" \
    --world_size "${WORLD_SIZE}"

EXIT_CODE=$?

echo ""
echo "=== DPO フルモデルマージ完了 ==="
echo "終了コード: ${EXIT_CODE}"
echo "終了時刻  : $(date)"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ マージ完了: ${OUT_DIR}"
    echo ""
    echo "推論テスト:"
    echo "  vllm serve ${OUT_DIR} --trust-remote-code"
else
    echo "✗ マージ失敗 (exit code: ${EXIT_CODE})"
fi

exit ${EXIT_CODE}
