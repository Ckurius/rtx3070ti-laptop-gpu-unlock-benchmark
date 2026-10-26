# Run summary – 20261026-125547_baseline_stock-repeat

- Phase: **baseline**
- gpu_unlock state: **stock**
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (0x24E010DE), VBIOS 94.04.7C.00.10, driver 595.99.02 (open)
- Power limit [N/A] (max 150.00 W), profile performance, AC True, Runtime D3 Disabled
- PyTorch 2.14.1+cu130, CUDA 13.0, cuDNN 92400
- Duration 21.7 min

| Test | Category | Expect | Median | Unit | CV % | n | Avg W | Max °C | Status |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| gemm_fp16_2048 | gemm | gain | 29.64 | TFLOPS | 1.6 | 5 | 100.5 | 53 | ok |
| gemm_bf16_2048 | gemm | gain | 30.77 | TFLOPS | 3.0 | 5 | 99.7 | 57 | ok |
| gemm_tf32_2048 | gemm | gain | 15.17 | TFLOPS | 2.3 | 5 | 100.5 | 59 | ok |
| gemm_fp32_2048 | gemm | control | 7.80 | TFLOPS | 1.5 | 5 | 102.3 | 62 | ok |
| gemm_int8_2048 | gemm | control | 23.67 | TOPS | 1.0 | 5 | 102.8 | 63 | ok |
| gemm_fp16_4096 | gemm | gain | 32.53 | TFLOPS | 1.3 | 5 | 102.6 | 64 | ok |
| gemm_bf16_4096 | gemm | gain | 33.14 | TFLOPS | 0.4 | 5 | 103.4 | 63 | ok |
| gemm_tf32_4096 | gemm | gain | 16.32 | TFLOPS | 1.0 | 5 | 103.4 | 64 | ok |
| gemm_fp32_4096 | gemm | control | 7.89 | TFLOPS | 0.8 | 5 | 103.6 | 64 | ok |
| gemm_int8_4096 | gemm | control | 23.39 | TOPS | 1.2 | 5 | 103.7 | 62 | ok |
| gemm_fp16_8192 | gemm | gain | 34.10 | TFLOPS | 0.6 | 5 | 103.5 | 65 | ok |
| gemm_bf16_8192 | gemm | gain | 33.77 | TFLOPS | 0.5 | 5 | 103.8 | 65 | ok |
| gemm_tf32_8192 | gemm | gain | 16.88 | TFLOPS | 0.8 | 5 | 103.6 | 66 | ok |
| gemm_fp32_8192 | gemm | control | 10.41 | TFLOPS | 1.1 | 5 | 103.5 | 64 | ok |
| gemm_int8_8192 | gemm | control | 22.87 | TOPS | 0.9 | 5 | 103.4 | 65 | ok |
| membw_copy_1gib | memory | control | 349.59 | GB/s | 0.0 | 5 | 93.8 | 62 | ok |
| sdpa_fwd_fp16 | kernel | gain | 31.65 | TFLOPS | 1.0 | 5 | 103.5 | 63 | ok |
| sdpa_fwd_bf16 | kernel | gain | 32.02 | TFLOPS | 0.9 | 5 | 103.0 | 63 | ok |
| sdpa_fwdbwd_bf16 | kernel | gain | 29.94 | TFLOPS | 1.4 | 5 | 102.2 | 63 | ok |
| conv3x3_fp16 | kernel | partial | 33.62 | TFLOPS | 0.4 | 5 | 103.4 | 63 | ok |
| conv3x3_bf16 | kernel | partial | 34.93 | TFLOPS | 0.2 | 5 | 103.9 | 63 | ok |
| llm_prefill_bf16_p512 | workload | gain | 14000.73 | tokens/s | 0.6 | 5 | 103.0 | 62 | ok |
| llm_prefill_bf16_p2048 | workload | gain | 15436.48 | tokens/s | 1.2 | 5 | 101.6 | 62 | ok |
| llm_decode_bf16_b1 | workload | info | 148.18 | tokens/s | 0.0 | 5 | 100.0 | 62 | ok |
| llm_decode_bf16_b16 | workload | partial | 1314.42 | tokens/s | 0.6 | 5 | 103.1 | 63 | ok |
| train_gpt124m_bf16 | workload | gain | 20251.30 | tokens/s | 0.6 | 5 | 101.0 | 62 | ok |
| train_gpt124m_fp16 | workload | gain | 19982.45 | tokens/s | 0.7 | 5 | 100.7 | 62 | ok |
| resnet50_train_bf16 | workload | partial | 477.28 | images/s | 0.8 | 5 | 102.2 | 62 | ok |
| resnet50_infer_fp16 | workload | partial | 1559.19 | images/s | 0.5 | 5 | 102.6 | 61 | ok |
| sustained_gemm_bf16_8192 | sustained | gain | 33.60 | TFLOPS | 0.4 | 49 | 103.5 | 67 | ok |
