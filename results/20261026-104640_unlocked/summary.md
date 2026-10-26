# Run summary – 20261026-104640_unlocked

- Phase: **unlocked**
- gpu_unlock state: **unlocked**
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (0x24E010DE), VBIOS 94.04.7C.00.10, driver 595.99.02 (open)
- Power limit [N/A] (max 150.00 W), profile performance, AC True, Runtime D3 Disabled
- PyTorch 2.14.1+cu130, CUDA 13.0, cuDNN 92400
- Duration 20.6 min

| Test | Category | Expect | Median | Unit | CV % | n | Avg W | Max °C | Status |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| gemm_fp16_2048 | gemm | gain | 41.82 | TFLOPS | 2.7 | 5 | 100.4 | 52 | ok |
| gemm_bf16_2048 | gemm | gain | 44.06 | TFLOPS | 2.6 | 5 | 101.3 | 56 | ok |
| gemm_tf32_2048 | gemm | gain | 23.06 | TFLOPS | 2.9 | 5 | 101.0 | 59 | ok |
| gemm_fp32_2048 | gemm | control | 7.59 | TFLOPS | 2.4 | 5 | 100.1 | 62 | ok |
| gemm_int8_2048 | gemm | control | 23.49 | TOPS | 2.4 | 5 | 101.5 | 63 | ok |
| gemm_fp16_4096 | gemm | gain | 49.87 | TFLOPS | 1.1 | 5 | 103.2 | 63 | ok |
| gemm_bf16_4096 | gemm | gain | 52.04 | TFLOPS | 1.8 | 5 | 103.2 | 62 | ok |
| gemm_tf32_4096 | gemm | gain | 24.84 | TFLOPS | 1.0 | 5 | 103.7 | 62 | ok |
| gemm_fp32_4096 | gemm | control | 7.90 | TFLOPS | 0.6 | 5 | 103.4 | 62 | ok |
| gemm_int8_4096 | gemm | control | 23.73 | TOPS | 0.9 | 5 | 104.2 | 62 | ok |
| gemm_fp16_8192 | gemm | gain | 52.31 | TFLOPS | 0.9 | 5 | 103.9 | 64 | ok |
| gemm_bf16_8192 | gemm | gain | 50.23 | TFLOPS | 1.1 | 5 | 103.4 | 64 | ok |
| gemm_tf32_8192 | gemm | gain | 25.46 | TFLOPS | 0.7 | 5 | 103.9 | 66 | ok |
| gemm_fp32_8192 | gemm | control | 10.26 | TFLOPS | 0.6 | 5 | 103.5 | 67 | ok |
| gemm_int8_8192 | gemm | control | 22.58 | TOPS | 0.7 | 5 | 103.3 | 66 | ok |
| membw_copy_1gib | memory | control | 349.46 | GB/s | 0.0 | 5 | 93.6 | 61 | ok |
| sdpa_fwd_fp16 | kernel | gain | 44.46 | TFLOPS | 0.6 | 5 | 103.5 | 63 | ok |
| sdpa_fwd_bf16 | kernel | gain | 45.89 | TFLOPS | 1.8 | 5 | 102.8 | 63 | ok |
| sdpa_fwdbwd_bf16 | kernel | gain | 41.68 | TFLOPS | 1.3 | 5 | 101.9 | 63 | ok |
| conv3x3_fp16 | kernel | partial | 50.64 | TFLOPS | 1.4 | 5 | 102.7 | 63 | ok |
| conv3x3_bf16 | kernel | partial | 52.45 | TFLOPS | 1.1 | 5 | 102.8 | 63 | ok |
| llm_prefill_bf16_p512 | workload | gain | 18910.33 | tokens/s | 0.6 | 5 | 102.5 | 62 | ok |
| llm_prefill_bf16_p2048 | workload | gain | 22199.93 | tokens/s | 1.0 | 5 | 103.1 | 62 | ok |
| llm_decode_bf16_b1 | workload | info | 149.72 | tokens/s | 0.0 | 5 | 100.8 | 62 | ok |
| llm_decode_bf16_b16 | workload | partial | 1524.76 | tokens/s | 0.3 | 5 | 103.9 | 62 | ok |
| train_gpt124m_bf16 | workload | gain | 26231.97 | tokens/s | 0.9 | 5 | 101.4 | 62 | ok |
| train_gpt124m_fp16 | workload | gain | 25312.60 | tokens/s | 0.8 | 5 | 101.8 | 62 | ok |
| resnet50_train_bf16 | workload | partial | 544.83 | images/s | 0.8 | 5 | 101.1 | 62 | ok |
| resnet50_infer_fp16 | workload | partial | 1814.97 | images/s | 0.4 | 5 | 102.2 | 61 | ok |
| sustained_gemm_bf16_8192 | sustained | gain | 49.95 | TFLOPS | 0.7 | 39 | 103.1 | 67 | ok |
