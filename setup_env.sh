#!/usr/bin/env bash
# Creates .venv with PyTorch (CUDA build), NVML bindings and plotting libs.
# Optional: WITH_SD=1 ./setup_env.sh  also installs diffusers for the Stable Diffusion test.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs
LOG="logs/setup_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== setup_env.sh $(date -Iseconds) on $(hostname), log: $LOG"

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found – install the NVIDIA driver first"; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if ! python3 -c "import ensurepip, venv" 2>/dev/null; then
  echo "python3-venv is missing, installing it (sudo)…"
  sudo apt-get install -y python3-venv
fi

[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install -q -U pip wheel

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "PyTorch with CUDA already present: $(python -c 'import torch; print(torch.__version__)')"
else
  ok=0
  for idx in cu130 cu129 cu128 cu126; do
    echo "== trying PyTorch wheels from index $idx"
    if python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$idx"; then ok=1; break; fi
  done
  [ "$ok" = 1 ] || { echo "ERROR: could not install PyTorch"; exit 1; }
fi
python -m pip install -q nvidia-ml-py numpy matplotlib
if [ "${WITH_SD:-0}" = "1" ]; then
  python -m pip install -q diffusers transformers accelerate safetensors
fi

python - <<'EOF'
import torch, pynvml
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "CUDA not available to PyTorch"
print("device:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
(a @ a).sum().item()
pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
print("NVML power reading:", pynvml.nvmlDeviceGetPowerUsage(h) / 1000, "W")
print("SETUP OK")
EOF
python -m pip freeze > "logs/pip-freeze_$(date +%Y%m%d-%H%M%S).txt"
