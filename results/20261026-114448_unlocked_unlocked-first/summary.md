# Run summary – 20261026-114448_unlocked_unlocked-first

- Phase: **unlocked**
- gpu_unlock state: **unlocked**
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (0x24E010DE), VBIOS 94.04.7C.00.10, driver 595.99.02 (open)
- Power limit [N/A] (max 150.00 W), profile performance, AC True, Runtime D3 Disabled
- PyTorch 2.14.1+cu130, CUDA 13.0, cuDNN 92400
- Duration 21.0 min

| Test | Category | Expect | Median | Unit | CV % | n | Avg W | Max °C | Status |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| gemm_fp16_2048 | gemm | gain | 41.91 | TFLOPS | 2.4 | 5 | 101.0 | 55 | ok |
| gemm_bf16_2048 | gemm | gain | 44.04 | TFLOPS | 3.3 | 5 | 100.7 | 57 | ok |
| gemm_tf32_2048 | gemm | gain | 22.68 | TFLOPS | 1.4 | 5 | 101.8 | 59 | ok |
| gemm_fp32_2048 | gemm | control | 7.66 | TFLOPS | 1.7 | 5 | 101.3 | 61 | ok |
| gemm_int8_2048 | gemm | control | 23.88 | TOPS | 2.2 | 5 | 101.8 | 62 | ok |
| gemm_fp16_4096 | gemm | gain | 49.98 | TFLOPS | 1.5 | 5 | 102.8 | 63 | ok |
| gemm_bf16_4096 | gemm | gain | 51.73 | TFLOPS | 1.2 | 5 | 102.8 | 63 | ok |
| gemm_tf32_4096 | gemm | gain | 24.95 | TFLOPS | 0.7 | 5 | 104.5 | 63 | ok |
| gemm_fp32_4096 | gemm | control | 7.97 | TFLOPS | 1.1 | 5 | 103.2 | 64 | ok |
| gemm_int8_4096 | gemm | control | 23.71 | TOPS | 0.7 | 5 | 104.1 | 62 | ok |
| gemm_fp16_8192 | gemm | gain | 52.00 | TFLOPS | 1.4 | 5 | 103.7 | 64 | ok |
| gemm_bf16_8192 | gemm | gain | 50.52 | TFLOPS | 0.6 | 5 | 104.4 | 64 | ok |
| gemm_tf32_8192 | gemm | gain | 25.56 | TFLOPS | 0.9 | 5 | 104.0 | 66 | ok |
| gemm_fp32_8192 | gemm | control | 10.37 | TFLOPS | 0.2 | 5 | 104.5 | 67 | ok |
| gemm_int8_8192 | gemm | control | 22.72 | TOPS | 0.4 | 5 | 104.1 | 66 | ok |
| membw_copy_1gib | memory | control | 349.54 | GB/s | 0.0 | 5 | 93.7 | 62 | ok |
| sdpa_fwd_fp16 | kernel | gain | 44.41 | TFLOPS | 0.4 | 5 | 103.7 | 63 | ok |
| sdpa_fwd_bf16 | kernel | gain | 45.79 | TFLOPS | 0.5 | 5 | 103.6 | 63 | ok |
| sdpa_fwdbwd_bf16 | kernel | gain | 41.90 | TFLOPS | 1.1 | 5 | 102.8 | 63 | ok |
| conv3x3_fp16 | kernel | partial | 50.70 | TFLOPS | 0.8 | 5 | 103.3 | 62 | ok |
| conv3x3_bf16 | kernel | partial | 52.49 | TFLOPS | 0.8 | 5 | 103.1 | 62 | ok |
| llm_prefill_bf16_p512 | workload | gain | 19051.63 | tokens/s | 0.5 | 5 | 103.2 | 62 | ok |
| llm_prefill_bf16_p2048 | workload | gain | 22329.36 | tokens/s | 1.8 | 5 | 102.2 | 62 | ok |
| llm_decode_bf16_b1 | workload | info | 149.72 | tokens/s | 0.1 | 5 | 100.5 | 62 | ok |
| llm_decode_bf16_b16 | workload | partial | 1527.82 | tokens/s | 0.0 | 5 | 104.9 | 63 | ok |
| train_gpt124m_bf16 | workload | gain | 26203.71 | tokens/s | 0.9 | 5 | 101.5 | 62 | ok |
| train_gpt124m_fp16 | workload | gain | 25391.63 | tokens/s | 1.0 | 5 | 101.4 | 62 | ok |
| resnet50_train_bf16 | workload | partial | 543.70 | images/s | 0.9 | 5 | 100.6 | 62 | ok |
| resnet50_infer_fp16 | workload | partial | 1817.21 | images/s | 0.5 | 5 | 102.5 | 62 | ok |
| sustained_gemm_bf16_8192 | sustained | gain | 50.29 | TFLOPS | 0.5 | 37 | 103.4 | 65 | ok |
