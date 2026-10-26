#!/usr/bin/env bash
# Test series 2: llama.cpp with real GGUF models (BF16, F16, Q8_0, Q4_K_M of the same model).
#
#   ./llama.sh setup       one-time: CUDA toolkit (if missing), build llama.cpp, download + convert + quantize model
#   ./llama.sh status      same preflight status as ./run.sh status
#   ./llama.sh smoke       ~3 min functional check (not comparable)
#   ./llama.sh baseline    full run on the stock card        (~25-35 min)
#   ./llama.sh unlocked    full run with gpu_unlock active    (~25-35 min)
#   ./llama.sh compare     report from the newest baseline + unlocked llama.cpp runs
#
# Settings (environment variables):
#   HF_MODEL=Qwen/Qwen2.5-3B-Instruct   source model (safetensors, BF16) on Hugging Face
#   QUANTS="BF16 F16 Q8_0 Q4_K_M"       GGUF variants to create and test
#   LLAMA_CPP_REF=<tag|commit>          pin llama.cpp to a specific version (default: newest master at setup time)
#   CUDA_PKG=cuda-toolkit-12-8          NVIDIA toolkit package used if no nvcc is installed
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$PWD"
HF_MODEL="${HF_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
QUANTS="${QUANTS:-BF16 F16 Q8_0 Q4_K_M}"
CUDA_PKG="${CUDA_PKG:-cuda-toolkit-12-8}"
LLAMA_DIR="$ROOT/third_party/llama.cpp"
MODEL_SLUG="$(basename "$HF_MODEL" | tr 'A-Z' 'a-z')"
MODEL_DIR="$ROOT/models"
GPU_UNLOCK_DIR="${GPU_UNLOCK_DIR:-$HOME/gpu-unlock}"
mkdir -p logs results/llamacpp reports/llamacpp "$MODEL_DIR" third_party
cmd="${1:-help}"; shift || true

oplog() {
  OPLOG="logs/${1}_$(date +%Y%m%d-%H%M%S).log"
  exec > >(tee -a "$OPLOG") 2>&1
  echo "== $1 $(date -Iseconds), log: $OPLOG"
}
step() { echo; echo "== [$(date +%H:%M:%S)] $*"; }

find_nvcc() {
  if command -v nvcc >/dev/null; then command -v nvcc; return; fi
  ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1
}

cuda_env() {
  local nvcc; nvcc="$(find_nvcc || true)"
  if [ -n "$nvcc" ]; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$nvcc")")")"
    export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi
}

venv() {
  [ -x .venv/bin/python ] || ./setup_env.sh
  # shellcheck disable=SC1091
  . .venv/bin/activate
}

bench() {
  venv
  cuda_env
  powerprofilesctl set performance 2>/dev/null || true
  local run=(python bench/llamacpp_suite.py --llama-bench "$LLAMA_DIR/build/bin/llama-bench"
             --models-dir "$MODEL_DIR" --model-slug "$MODEL_SLUG" --quants "$QUANTS"
             --gpu-unlock-dir "$GPU_UNLOCK_DIR" "$@")
  if command -v systemd-inhibit >/dev/null; then
    systemd-inhibit --what=idle:sleep --who="3070ti-unlock" --why="llama.cpp benchmark running" "${run[@]}"
  else
    "${run[@]}"
  fi
}

case "$cmd" in
  setup)
    oplog llamacpp-setup
    step "1/6 build dependencies"
    need=()
    for p in build-essential cmake git; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
    [ ${#need[@]} -eq 0 ] || sudo apt-get install -y "${need[@]}"
    cmake --version | head -1

    step "2/6 CUDA toolkit (nvcc)"
    if [ -z "$(find_nvcc || true)" ]; then
      echo "No nvcc found -> installing $CUDA_PKG from NVIDIA's apt repository (toolkit only, no driver)."
      tmp="$(mktemp -d)"
      wget -q -O "$tmp/cuda-keyring.deb" \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
      sudo dpkg -i "$tmp/cuda-keyring.deb"
      # Low priority for the NVIDIA repo: it can never replace Ubuntu's driver packages on upgrades.
      printf 'Package: *\nPin: origin developer.download.nvidia.com\nPin-Priority: 100\n' |
        sudo tee /etc/apt/preferences.d/nvidia-cuda-repo-low-priority >/dev/null
      sudo apt-get update
      sudo apt-get install -y --no-install-recommends "$CUDA_PKG"
    fi
    cuda_env
    nvcc --version | tail -2

    step "3/6 llama.cpp source"
    if [ ! -d "$LLAMA_DIR/.git" ]; then
      git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    fi
    if [ -n "${LLAMA_CPP_REF:-}" ]; then
      git -C "$LLAMA_DIR" fetch --tags -q origin
      git -C "$LLAMA_DIR" checkout -q "$LLAMA_CPP_REF"
    fi
    git -C "$LLAMA_DIR" log -1 --format='llama.cpp commit %h (%cs) %s'

    step "4/6 build llama.cpp with CUDA for sm_86 (this takes a while)"
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
      -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" --target llama-bench llama-quantize
    "$LLAMA_DIR/build/bin/llama-bench" --help | grep -E "flash-attn|repetitions" || true

    step "5/6 download $HF_MODEL and convert to GGUF"
    if [ ! -x .venv-convert/bin/python ]; then
      python3 -m venv .venv-convert
      .venv-convert/bin/python -m pip install -q -U pip wheel
      .venv-convert/bin/python -m pip install -q -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"
      .venv-convert/bin/python -m pip install -q "huggingface_hub>=0.34,<1.0"
    fi
    src="$MODEL_DIR/hf/$(basename "$HF_MODEL")"
    .venv-convert/bin/python - "$HF_MODEL" "$src" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1], sys.argv[2]
p = snapshot_download(repo, local_dir=dst,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "tokenizer*"])
print("downloaded to", p)
EOF
    base="$MODEL_DIR/${MODEL_SLUG}-BF16.gguf"
    [ -s "$base" ] || .venv-convert/bin/python "$LLAMA_DIR/convert_hf_to_gguf.py" "$src" --outtype bf16 --outfile "$base"

    step "6/6 create variants: $QUANTS"
    for q in $QUANTS; do
      out="$MODEL_DIR/${MODEL_SLUG}-${q}.gguf"
      [ -s "$out" ] && { echo "exists: $out"; continue; }
      case "$q" in
        BF16) ;;
        F16)  .venv-convert/bin/python "$LLAMA_DIR/convert_hf_to_gguf.py" "$src" --outtype f16 --outfile "$out" ;;
        *)    "$LLAMA_DIR/build/bin/llama-quantize" "$base" "$out" "$q" ;;
      esac
    done
    (cd "$MODEL_DIR" && sha256sum ./*.gguf | tee SHA256SUMS)
    ls -lh "$MODEL_DIR"/*.gguf
    echo; echo ">>> Setup finished. Next: ./llama.sh smoke"
    ;;

  status)   python3 bench/sysinfo.py status ;;
  smoke)    bench --phase auto --smoke "$@" ;;
  baseline) bench --phase baseline "$@" ;;
  unlocked) bench --phase unlocked "$@" ;;
  compare)  venv; python bench/llamacpp_compare.py "$@" ;;
  *)        sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
