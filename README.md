# RTX 3070 Ti Laptop GPU – gpu-unlock benchmark

A reproducible before/after benchmark of [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) on an
NVIDIA GeForce RTX 3070 Ti **Laptop** GPU (GA104, `10de:24a0`). gpu-unlock is a Linux kernel module that removes the
half-rate limit on FP16/BF16/TF32 tensor-core math with FP32 accumulation on GeForce Ampere cards. At the time of
writing, no laptop GPU had been reported in its compatibility list.

The suite runs an identical test list twice, once on the stock card (**baseline**) and once with gpu_unlock active
(**unlocked**). It then produces a report with validity checks, all raw logs and SHA-256 checksums.

## What is measured

| Category | Tests | Expected effect |
|---|---|---|
| Correctness | FP16/BF16/TF32/FP32 2048³ GEMM output hash + error vs FP64 (also verifies the TF32 switch) | identical |
| GEMM | FP16, BF16, TF32 at 2048/4096/8192 | **gain** |
| GEMM controls | FP32 (CUDA cores, TF32 off), INT8 (`torch._int_mm`) | none |
| Memory | 1 GiB device copy | none (control) |
| Kernels | SDPA attention fwd / fwd+bwd, 3×3 convolution (cuDNN) | gain / partial |
| LLM | prompt processing 512/2048 tokens, token generation batch 1 and 16 (synthetic ~0.95 B Llama-style model) | gain / info / partial |
| Training | GPT-style 124 M (BF16 and FP16), ResNet-50 (BF16) | gain / partial |
| Inference | ResNet-50 FP16; optional Stable Diffusion via diffusers | partial / gain |
| Sustained | BF16 8192³ GEMM for 180 s in 5 s windows | gain; shows power/thermal limits |

Every throughput test is warmed up, then measured as 5 samples of about 2 s each with CUDA events. Before each test
the GPU idles until it is at or below 55 °C. Power, clocks, temperature and NVML throttle reasons are logged at 10 Hz.
Control tests, which must not change, confirm that the two boots differ only in the unlock.

## Requirements

- Ubuntu 24.04 (or similar), NVIDIA driver ≥ 580, Python 3.10+
- Power adapter connected, platform profile `performance`
- NVIDIA Runtime D3 disabled (`./run.sh prepare`). Otherwise the GPU powers off when idle and loses the unlock.

## Usage

```bash
./setup_env.sh            # venv + PyTorch (CUDA) + nvidia-ml-py + matplotlib; WITH_SD=1 adds diffusers
./run.sh prepare          # one-time: disable Runtime D3, performance profile -> reboot
./run.sh status           # shows unlock state, profile, AC, Runtime D3
./run.sh smoke            # 2-3 min functional check of every test
./run.sh baseline         # stock card, ~30-40 min
./run.sh install-unlock   # installs gpu-unlock + kernel parameter -> reboot
./run.sh unlocked         # unlocked card, ~30-40 min
./run.sh compare          # report from the newest baseline + unlocked run
./run.sh remove-unlock    # back to stock -> reboot
```

Useful options for `baseline`/`unlocked`: `--samples 7`, `--sustained-sec 300`, `--only 'gemm_.*_8192'`,
`--skip resnet`, `--sd-model stabilityai/sd-turbo`, `--label <note>`. See `python bench/suite.py --help`.

## Output

```
results/<timestamp>_<phase>/
  run.log          human-readable log
  env.json, env/   environment summary and verbatim command outputs (nvidia-smi -q, lspci, kernel log …)
  results.jsonl    one JSON record per test: all samples, statistics, telemetry, parameters
  results.csv      one row per test
  telemetry.csv    10 Hz GPU telemetry tagged with the running test
  sustained_*.csv  per-window timeline of the sustained test
  summary.md       per-run table
  manifest.json    SHA-256 of every file
reports/<timestamp>_<baseline>_vs_<unlocked>/
  report.md        the publishable report (validity checks, results, charts)
  compat_report.md pre-filled gpu-unlock compatibility report
  comparison.csv / comparison.json / compare.log / charts/*.png
logs/              setup and admin actions (prepare, install-unlock, remove-unlock)
```

## Test series 2: llama.cpp with GGUF models

Most people run quantized GGUF models with llama.cpp, Ollama or LM Studio, not BF16 PyTorch. This series measures
that case with `llama-bench`. It uses one source model (default `Qwen/Qwen2.5-3B-Instruct`), converted once to BF16
GGUF and quantised from that file into F16, Q8_0 and Q4_K_M, so every variant is the same network and fits into 8 GB.

In the current llama.cpp CUDA backend each weight format takes a different path, which predicts the result:

| Weights | Prompt processing path | Expected effect of the unlock |
|---|---|---|
| BF16 | cuBLAS, BF16 inputs, **FP32 accumulation** | large (the unlocked path) |
| F16 | cuBLAS, FP16 inputs, FP16 accumulation | small (not rate-limited on GeForce) |
| Q8_0, Q4_K_M | MMQ kernels, INT8 tensor cores | small |
| any (token generation) | memory-bandwidth bound | none (control) |

Each weight format is tested for prompt processing (512 and 2048 tokens) and token generation (128 tokens), with
flash attention off and on. Each test has 7 repetitions, preceded by the same thermal cool-down and telemetry as series 1.

```bash
./llama.sh setup      # CUDA toolkit if missing, llama.cpp build (sm_86), model download/convert/quantize (~30-60 min, ~35 GB disk)
./llama.sh smoke      # ~3 min functional check
./llama.sh baseline   # stock card, ~25-35 min
./llama.sh unlocked   # gpu_unlock active, ~25-35 min
./llama.sh compare    # report in reports/llamacpp/
```

Use a different model with `HF_MODEL=org/name ./llama.sh setup`. Choose variants with `QUANTS="BF16 Q4_K_M"`. Pin
llama.cpp with `LLAMA_CPP_REF=<tag>`. Results go to `results/llamacpp/<timestamp>_<phase>/`. The raw `llama-bench` JSON
of every test is kept in `raw/`.

## Disclaimer

gpu-unlock is experimental third-party software that writes undocumented GPU registers. This repository only
measures its effect. Use at your own risk.

## Results

**Series 1 – HP Omen 16 (RTX 3070 Ti Laptop, 10de:24e0), both phases at the same ~103 W power cap:**

| Workload | Stock | Unlocked | Change |
|---|---:|---:|---:|
| BF16 GEMM 8192³ (FP32 acc.) | 33.5 TFLOPS | 50.2 TFLOPS | **+50 %** |
| FP16 GEMM 8192³ (FP32 acc.) | 34.0 TFLOPS | 52.3 TFLOPS | **+54 %** |
| TF32 GEMM 8192³ | 16.9 TFLOPS | 25.5 TFLOPS | **+51 %** |
| LLM prompt processing, 2048 tokens (BF16) | 15,452 tok/s | 22,200 tok/s | **+44 %** |
| GPT-124M training (BF16) | 20,318 tok/s | 26,232 tok/s | **+29 %** |
| LLM token generation, batch 1 | 148 tok/s | 150 tok/s | +1 % |

All validity checks passed. Controls (FP32, INT8, memory bandwidth) unchanged within ±2 %, outputs bit-identical.
Same power draw, lower clocks, ~48 % more work per watt.
Full report: [series 1](reports/20261026-110806_20261026-100526_baseline_vs_20261026-104640_unlocked/report.md)

- Series 2 (llama.cpp / GGUF): coming soon
