#!/bin/bash
# ============================================================
# Qwen3-Next-80B-A3B 統合環境セットアップ
#
# vLLM serve + FSDP trainer を同一 conda 環境で動かす。
# Qwen3.5-397B では transformers 5.x と vLLM nightly の依存衝突で
# 2つに分けていたが、Qwen3-Next は transformers 4.57+ で動くので
# 1つの環境で十分。
#
# 必要: CUDA 12.4+, miniconda/conda
#
# 使い方:
#   bash env.sh
#
# 所要時間: 約10-15分
# ============================================================

set -euo pipefail

ENV_NAME="qwen3next"
PYTHON_VERSION="3.11"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"

echo "================================================================"
echo "  Qwen3-Next-80B-A3B 統合環境セットアップ"
echo "  env name : ${ENV_NAME}"
echo "  python   : ${PYTHON_VERSION}"
echo "  conda    : ${CONDA_BASE}"
echo "================================================================"

# ============================================================
# 1) conda env 作成
# ============================================================
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/7] env '${ENV_NAME}' already exists. Recreate? [y/N]"
    read -r ans
    if [[ "${ans}" =~ ^[Yy]$ ]]; then
        echo "[1/7] Removing existing env..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "[1/7] Reusing existing env."
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/7] Creating conda env: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[1/7] Python: $(python --version)"
echo "[1/7] pip: $(pip --version)"

# ============================================================
# 2) PyTorch (CUDA 12.4)
# ============================================================
echo ""
echo "[2/7] Installing PyTorch (CUDA 12.4)..."

# PyTorch 2.6+ with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# ============================================================
# 3) vLLM
# ============================================================
echo ""
echo "[3/7] Installing vLLM..."

# vLLM >= 0.10.2 required for Qwen3-Next
# nightly has best Qwen3-Next support (GDN kernels, etc.)
# まず stable を試し、ダメなら nightly にフォールバック
pip install vllm || {
    echo "[3/7] Stable vLLM failed, trying nightly..."
    pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
}

python -c "
import vllm
print(f'  vLLM: {vllm.__version__}')
"

# ============================================================
# 4) transformers (Qwen3-Next support)
# ============================================================
echo ""
echo "[4/7] Installing transformers (Qwen3-Next support)..."

# Qwen3-Next の config.json は transformers_version: 4.57.0.dev0
# まず最新 stable を試し、Qwen3NextForCausalLM が使えるか確認
pip install --upgrade transformers

# Qwen3-Next モデルクラスの存在確認
QWEN3NEXT_OK=$(python -c "
try:
    from transformers import AutoConfig
    c = AutoConfig.for_model('qwen3_next')
    print('ok')
except Exception:
    print('ng')
" 2>/dev/null)

if [ "${QWEN3NEXT_OK}" != "ok" ]; then
    echo "[4/7] Stable transformers lacks Qwen3-Next support, installing from main..."
    pip install --upgrade git+https://github.com/huggingface/transformers.git@main
fi

python -c "
import transformers
print(f'  transformers: {transformers.__version__}')
try:
    from transformers import AutoConfig
    AutoConfig.for_model('qwen3_next')
    print('  Qwen3-Next support: OK')
except Exception as e:
    print(f'  Qwen3-Next support: MISSING ({e})')
"

# ============================================================
# 5) TRL (WeightSyncWorkerExtension)
# ============================================================
echo ""
echo "[5/7] Installing TRL..."

pip install trl

# WeightSyncWorkerExtension の import 確認
python -c "
import trl
print(f'  TRL: {trl.__version__}')
try:
    from trl.scripts.vllm_serve import WeightSyncWorkerExtension
    print('  WeightSyncWorkerExtension: OK')
except ImportError as e:
    print(f'  WeightSyncWorkerExtension: FAILED ({e})')
    print('  → vllm_ascend stub may be needed (handled in run script)')
"

# ============================================================
# 6) accelerate + その他
# ============================================================
echo ""
echo "[6/7] Installing accelerate and dependencies..."

pip install --upgrade accelerate
pip install requests fastapi uvicorn pydantic
pip install wandb  # Optional: W&B logging

python -c "
import accelerate
print(f'  accelerate: {accelerate.__version__}')
"

# ============================================================
# 7) vllm_ascend stub (WeightSyncWorkerExtension 用)
# ============================================================
echo ""
echo "[7/7] Creating vllm_ascend stub (if needed)..."

python << 'STUB_SCRIPT'
import os
import site

try:
    sp = site.getsitepackages()[0]
except Exception:
    import sys
    sp = os.path.join(os.path.dirname(sys.executable), "..", "lib",
                      f"python{sys.version_info.major}.{sys.version_info.minor}",
                      "site-packages")

pkg = os.path.join(sp, "vllm_ascend", "distributed", "device_communicators")
os.makedirs(pkg, exist_ok=True)

for d in [
    os.path.join(sp, "vllm_ascend"),
    os.path.join(sp, "vllm_ascend", "distributed"),
    pkg,
]:
    init = os.path.join(d, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()

pyhccl = os.path.join(pkg, "pyhccl.py")
if not os.path.exists(pyhccl):
    with open(pyhccl, "w") as f:
        f.write("class PyHcclCommunicator: pass\n")
    print("  Created vllm_ascend stub")
else:
    print("  vllm_ascend stub already exists")
STUB_SCRIPT

# 最終確認
echo ""
echo "================================================================"
echo "  WeightSyncWorkerExtension 最終確認"
echo "================================================================"

python -c "
from trl.scripts.vllm_serve import WeightSyncWorkerExtension
print('  ✅ WeightSyncWorkerExtension import OK')
" 2>&1 || {
    echo "  ⚠️  WeightSyncWorkerExtension import failed"
    echo "  手動で確認してください"
}

# ============================================================
# Summary
# ============================================================
echo ""
echo "================================================================"
echo "  セットアップ完了!"
echo "================================================================"
echo ""

python << 'SUMMARY'
import sys
print(f"  Python       : {sys.version.split()[0]}")

import torch
print(f"  PyTorch      : {torch.__version__}")
print(f"  CUDA         : {torch.version.cuda}")

import vllm
print(f"  vLLM         : {vllm.__version__}")

import transformers
print(f"  transformers : {transformers.__version__}")

import trl
print(f"  TRL          : {trl.__version__}")

import accelerate
print(f"  accelerate   : {accelerate.__version__}")

# Qwen3-Next check
try:
    from transformers import AutoConfig
    AutoConfig.for_model("qwen3_next")
    print(f"  Qwen3-Next   : ✅ supported")
except Exception:
    print(f"  Qwen3-Next   : ❌ not supported")

# WeightSync check
try:
    from trl.scripts.vllm_serve import WeightSyncWorkerExtension
    print(f"  WeightSync   : ✅ OK")
except Exception:
    print(f"  WeightSync   : ❌ failed")
SUMMARY

echo ""
echo "  使い方:"
echo "    conda activate ${ENV_NAME}"
echo "    sbatch run_qwen3next_fullparam_grpo.sh"
echo ""
echo "  シェルスクリプトの CONDA_ENV_VLLM / CONDA_ENV_TRAINER を"
echo "  両方とも以下に変更:"
echo "    ${CONDA_BASE}/envs/${ENV_NAME}"
echo ""
echo "================================================================"
