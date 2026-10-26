# Compatibility report (paste into the gpu-unlock issue form)

**Title:** Compatibility report: NVIDIA GeForce RTX 3070 Ti Laptop GPU (10de:24e0)

**Result:** Unlocked, and FP16/BF16 got faster

**Card:** NVIDIA GeForce RTX 3070 Ti Laptop GPU 8 GB in HP Omen 16-k0620ng

**PCI IDs:**
```text
0000:01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA104M [Geforce RTX 3070 Ti Laptop GPU] [10de:24e0] (rev a1)
0000:01:00.1 Audio device [0403]: NVIDIA Corporation GA104 High Definition Audio Controller [10de:228b] (rev a1)
```

**VBIOS version:** 00000000:01:00.0, 94.04.7C.00.10

**Is the card modified?** Stock

**Distribution:** Ubuntu 24.04.5 LTS

**Kernel:** 7.0.0-38-generic

**NVIDIA driver:** 595.99.02, open kernel modules

**Initramfs generator:** initramfs-tools

**gpu-unlock version and install method:** af28cbb, make install

**Kernel command line:** `BOOT_IMAGE=/boot/vmlinuz-7.0.0-38-generic root=UUID=0e3f2d5a-c5b3-427e-8a2f-5edc47b790fd ro quiet splash gpu_unlock.targets=0000:01:00.0 vt.handoff=7`

**Setup:**
```text
Laptop (Optimus, iGPU drives the display), single GPU, enforced power limit 105 W (max 150.00 W), Dynamic Boost Supported, nvidia-powerd active, NVIDIA Runtime D3 Disabled, Secure Boot: SecureBoot disabled
```

**Kernel log:**
```text
2026-10-26T10:43:46+01:00 Kiubu kernel: Command line: BOOT_IMAGE=/boot/vmlinuz-7.0.0-38-generic root=UUID=0e3f2d5a-c5b3-427e-8a2f-5edc47b790fd ro quiet splash gpu_unlock.targets=0000:01:00.0 vt.handoff=7
2026-10-26T10:43:46+01:00 Kiubu kernel: Kernel command line: BOOT_IMAGE=/boot/vmlinuz-7.0.0-38-generic root=UUID=0e3f2d5a-c5b3-427e-8a2f-5edc47b790fd ro quiet splash gpu_unlock.targets=0000:01:00.0 vt.handoff=7
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: loading out-of-tree module taints kernel.
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: module verification failed: signature and/or required key missing - tainting kernel
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: ROM chain: pos=0x0 data=0x52494350 bytes=65024 type=0x0 last=0
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: ROM chain: pos=0xfe00 data=0x52494350 bytes=83968 type=0x3 last=0
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: ROM chain: pos=0x24600 data=0x5344504e bytes=22016 type=0xe0 last=0
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: ROM chain: pos=0x29c00 data=0x5344504e bytes=429568 type=0xe0 last=128
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: ROM expansion: base=0xfe00 ext=0x24600 adjust=0x14800
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: ROM BIT: offset=0x1b0 size=12 stride=6 tokens=18 adjust=0x14800
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: ROM ucode table: offset=0x92887 version=1 header=6 stride=6 entries=16
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: FWSEC descriptor: offset=0x411ac ROM_bytes=1048576 header=0x4ac0301
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: FWSEC geometry: image_off=0x41658 size=0xe900 imem=0xe100 va=0x0 pa=0x0 dmem=0x800 pa=0x0 pkc=0x5a4 interface=0x1c engine=0x400 ucode=9 signatures=3 versions=0x7 bytes=1152
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: FWSEC interface: offset=0x1c version=1 header=4 stride=8 entries=2 FRTS=0x1ffe00000
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: FWSEC mapper: version=3 size=64 cmd_offset=0x7c0 cmd_size=64
2026-10-26T10:43:46+01:00 Kiubu kernel: gpu_unlock: FWSEC signature selection: versions=0x7 fuse_version=2 sig_bytes=384
2026-10-26T10:43:46+01:00 Kiubu kernel: pci 0000:01:00.0: gpu_unlock: unlocked
```

**Benchmark before and after:**
```text
torch 2.14.1+cu130 cuBLAS (CUDA 13.0), 8192³ dense GEMM, FP32 accumulation, median of 5 samples, ~103 W measured
```

| Input | Before | After | Change |
| ----- | ------ | ----- | ------ |
| FP16 | 34.03 TFLOPS | 52.31 TFLOPS | +53.7 % |
| BF16 | 33.48 TFLOPS | 50.23 TFLOPS | +50.0 % |
| TF32 | 16.87 TFLOPS | 25.46 TFLOPS | +50.9 % |
| LLM prefill 2048 tok | 15451.6 tokens/s | 22199.9 tokens/s | +43.7 % |
| LLM decode b1 | 148.2 tokens/s | 149.7 tokens/s | +1.1 % |
| GPT-124M train BF16 | 20318.1 tokens/s | 26232.0 tokens/s | +29.1 % |

Full report and raw logs: <link to your repository>
