# Run summary – 20261026-100526_baseline

- Phase: **baseline**
- gpu_unlock state: **stock**
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (0x24E010DE), VBIOS 94.04.7C.00.10, driver 595.99.02 (open)
- Power limit [N/A] (max 150.00 W), profile performance, AC True, Runtime D3 Disabled
- PyTorch 2.14.1+cu130, CUDA 13.0, cuDNN 92400
- Duration 21.6 min

| Test | Category | Expect | Median | Unit | CV % | n | Avg W | Max °C | Status |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| gemm_fp16_2048 | gemm | gain | 29.68 | TFLOPS | 1.8 | 5 | 100.3 | 50 | ok |
| gemm_bf16_2048 | gemm | gain | 30.69 | TFLOPS | 2.2 | 5 | 101.1 | 54 | ok |
| gemm_tf32_2048 | gemm | gain | 15.24 | TFLOPS | 1.9 | 5 | 101.0 | 57 | ok |
| gemm_fp32_2048 | gemm | control | 7.75 | TFLOPS | 1.3 | 5 | 101.8 | 59 | ok |
| gemm_int8_2048 | gemm | control | 23.85 | TOPS | 1.3 | 5 | 102.7 | 61 | ok |
| gemm_fp16_4096 | gemm | gain | 32.60 | TFLOPS | 0.4 | 5 | 103.9 | 63 | ok |
| gemm_bf16_4096 | gemm | gain | 33.17 | TFLOPS | 0.6 | 5 | 103.6 | 63 | ok |
| gemm_tf32_4096 | gemm | gain | 16.25 | TFLOPS | 0.4 | 5 | 103.5 | 64 | ok |
| gemm_fp32_4096 | gemm | control | 7.91 | TFLOPS | 1.2 | 5 | 103.3 | 62 | ok |
| gemm_int8_4096 | gemm | control | 23.46 | TOPS | 1.0 | 5 | 103.8 | 61 | ok |
| gemm_fp16_8192 | gemm | gain | 34.03 | TFLOPS | 0.4 | 5 | 103.7 | 64 | ok |
| gemm_bf16_8192 | gemm | gain | 33.48 | TFLOPS | 0.5 | 5 | 103.0 | 65 | ok |
| gemm_tf32_8192 | gemm | gain | 16.87 | TFLOPS | 0.3 | 5 | 104.5 | 66 | ok |
| gemm_fp32_8192 | gemm | control | 10.26 | TFLOPS | 0.9 | 5 | 103.3 | 67 | ok |
| gemm_int8_8192 | gemm | control | 22.85 | TOPS | 0.8 | 5 | 103.5 | 65 | ok |
| membw_copy_1gib | memory | control | 349.57 | GB/s | 0.0 | 5 | 93.8 | 62 | ok |
| sdpa_fwd_fp16 | kernel | gain | 31.61 | TFLOPS | 0.6 | 5 | 103.6 | 63 | ok |
| sdpa_fwd_bf16 | kernel | gain | 32.43 | TFLOPS | 0.8 | 5 | 103.9 | 63 | ok |
| sdpa_fwdbwd_bf16 | kernel | gain | 30.00 | TFLOPS | 0.9 | 5 | 103.3 | 63 | ok |
| conv3x3_fp16 | kernel | partial | 33.65 | TFLOPS | 0.3 | 5 | 103.8 | 63 | ok |
| conv3x3_bf16 | kernel | partial | 34.98 | TFLOPS | 0.5 | 5 | 103.9 | 63 | ok |
| llm_prefill_bf16_p512 | workload | gain | 13990.88 | tokens/s | 0.6 | 5 | 102.6 | 62 | ok |
| llm_prefill_bf16_p2048 | workload | gain | 15451.60 | tokens/s | 0.5 | 5 | 102.9 | 62 | ok |
| llm_decode_bf16_b1 | workload | info | 148.16 | tokens/s | 0.0 | 5 | 99.6 | 61 | ok |
| llm_decode_bf16_b16 | workload | partial | 1319.47 | tokens/s | 0.5 | 5 | 103.4 | 62 | ok |
| train_gpt124m_bf16 | workload | gain | 20318.14 | tokens/s | 0.4 | 5 | 101.2 | 60 | ok |
| train_gpt124m_fp16 | workload | gain | 20077.78 | tokens/s | 0.5 | 5 | 101.8 | 61 | ok |
| resnet50_train_bf16 | workload | partial | 476.49 | images/s | 0.9 | 5 | 101.5 | 61 | ok |
| resnet50_infer_fp16 | workload | partial | 1553.12 | images/s | 0.9 | 5 | 101.5 | 61 | ok |
| sustained_gemm_bf16_8192 | sustained | gain | 33.66 | TFLOPS | 0.4 | 39 | 103.1 | 66 | ok |
