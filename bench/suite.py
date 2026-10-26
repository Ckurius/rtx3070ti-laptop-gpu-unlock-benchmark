#!/usr/bin/env python3
"""RTX 30 (Ampere) gpu-unlock benchmark suite.

Runs an identical, fixed set of tests in two phases ("baseline" = stock,
"unlocked" = gpu_unlock active) and writes clean, machine-readable logs:

  results/<timestamp>_<phase>/
    run.log            human-readable log of everything that happened
    env.json, env/     environment summary + verbatim raw command outputs
    results.jsonl      one JSON record per test (all samples, stats, telemetry)
    results.csv        one flat row per test (for spreadsheets)
    telemetry.csv      GPU power/clock/temp/throttle at 10 Hz, tagged by test
    sustained_*.csv    per-window timeline of the long-running test
    summary.md         per-run summary table
    manifest.json      SHA-256 of every file (integrity for publication)

Use compare.py to turn two runs into a publishable report.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (SUITE_VERSION, JsonlWriter, Telemetry, describe, now_iso,  # noqa: E402
                    setup_logging, sh, sha256_file)
import sysinfo  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    print("PyTorch is missing. Run ./setup_env.sh first.", file=sys.stderr)
    sys.exit(2)

DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# expectation labels, used by compare.py to validate the measurement
GAIN = "gain"          # path the unlock targets: FP16/BF16/TF32 tensor math, FP32 accumulate
PARTIAL = "partial"    # mixed workload, part of the time in the unlocked path
CONTROL = "control"    # must NOT change (FP32 CUDA cores, INT8, memory bandwidth)
INFO = "info"          # correctness / informational


# =========================================================================== models

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        r = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * r.to(x.dtype)) * self.weight


class Block(nn.Module):
    """Llama-style decoder block (RMSNorm, fused QKV, SDPA attention, SwiGLU)."""

    def __init__(self, dim: int, heads: int, ffn: int):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.n1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.n2 = RMSNorm(dim)
        self.w13 = nn.Linear(dim, 2 * ffn, bias=False)
        self.w2 = nn.Linear(ffn, dim, bias=False)

    def forward(self, x, cache=None, pos: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        if cache is not None:
            kc, vc = cache
            kc[:, :, pos:pos + T] = k
            vc[:, :, pos:pos + T] = v
            k, v = kc[:, :, :pos + T], vc[:, :, :pos + T]
        # Causal only for prefill/training (T > 1 starting at pos 0); decode uses T == 1.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, C))
        g, u = self.w13(self.n2(x)).chunk(2, dim=-1)
        return x + self.w2(F.silu(g) * u)


class Decoder(nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int, heads: int, ffn: int, tie: bool = False):
        super().__init__()
        self.cfg = dict(vocab=vocab, dim=dim, layers=layers, heads=heads, ffn=ffn, tied_embeddings=tie)
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, ffn) for _ in range(layers))
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        if tie:  # GPT-2 style; not used with build_decoder_lowp (to_empty would untie)
            self.head.weight = self.emb.weight

    def forward(self, idx, caches=None, pos: int = 0, last_only: bool = False):
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, caches[i] if caches is not None else None, pos)
        x = self.norm(x)
        if last_only:
            x = x[:, -1:]
        return self.head(x)

    def matmul_params(self) -> int:
        """Parameters that take part in a matmul per token (excludes embedding lookup)."""
        n = sum(p.numel() for name, p in self.named_parameters()
                if p.dim() == 2 and not name.startswith("emb."))
        return n


def build_decoder_lowp(cfg: dict, device, dtype) -> Decoder:
    """Build directly in low precision on the GPU without an FP32 copy (saves VRAM)."""
    with torch.device("meta"):
        m = Decoder(**cfg)
    m = m.to(dtype).to_empty(device=device)
    g = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.02, generator=g)
            else:
                p.fill_(1.0)
    return m.eval()


LLM_CFG = dict(vocab=32000, dim=2048, layers=16, heads=16, ffn=5632)        # ~0.95 B params
GPT_CFG = dict(vocab=50304, dim=768, layers=12, heads=12, ffn=2048)         # ~124 M params


# =========================================================================== harness

@dataclass
class Prepared:
    fn: Callable[[], object]
    work: float                       # work units per fn() call (FLOP, bytes, tokens, images ...)
    scale: float                      # metric = work * calls / seconds / scale
    extra: dict = field(default_factory=dict)


@dataclass
class TestSpec:
    id: str
    category: str
    expectation: str
    unit: str
    desc: str
    prepare: Callable[[object], Prepared] | None = None   # prepare(batch) -> Prepared
    batches: list = field(default_factory=lambda: [None])
    kind: str = "throughput"          # throughput | sustained | correctness


class Ctx:
    def __init__(self, args, log, tele: Telemetry, jsonl: JsonlWriter, run_dir: str):
        self.args, self.log, self.tele, self.jsonl, self.run_dir = args, log, tele, jsonl, run_dir
        self.dev = torch.device("cuda:0")
        self.csv_rows: list[dict] = []


def sync():
    torch.cuda.synchronize()


def timed(fn, calls: int) -> float:
    """Run fn `calls` times, return GPU wall time in seconds (CUDA events)."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(calls):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / 1000.0


def set_tf32(on: bool) -> None:
    """Toggle TF32 for cuBLAS matmuls (new fp32_precision API if present, legacy flag otherwise)."""
    m = torch.backends.cuda.matmul
    if hasattr(m, "fp32_precision"):
        m.fp32_precision = "tf32" if on else "ieee"
    else:
        m.allow_tf32 = on


def free_gpu():
    set_tf32(False)   # every test starts from the same flags
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def cooldown(ctx: Ctx, test_id: str) -> float | None:
    """Thermal equalisation: identical in both phases, so tests start from comparable states."""
    a = ctx.args
    ctx.tele.set_tag(f"cooldown:{test_id}")
    t0 = time.time()
    time.sleep(a.cooldown_min)
    temp = ctx.tele.current_temp()
    while temp is not None and temp > a.cooldown_temp and time.time() - t0 < a.cooldown_max:
        time.sleep(1.0)
        temp = ctx.tele.current_temp()
    waited = time.time() - t0
    if temp is not None and temp > a.cooldown_temp:
        ctx.log.warning("%s: start temperature %s °C above target %s °C after %.0f s cooldown",
                        test_id, temp, a.cooldown_temp, waited)
    ctx.log.debug("%s: cooldown %.1f s, start temp %s °C", test_id, waited, temp)
    return temp


def measure(ctx: Ctx, p: Prepared, test_id: str) -> tuple[list[float], int, dict, float]:
    a = ctx.args
    # warm-up (also triggers cuDNN autotuning / lazy init)
    ctx.tele.set_tag(f"warmup:{test_id}")
    t_end = time.time() + a.warmup_sec
    n_warm = 0
    while time.time() < t_end or n_warm < 2:
        p.fn()
        n_warm += 1
    sync()
    # calibrate calls per sample so each sample lasts ~target_sec
    t1 = timed(p.fn, 1)
    t3 = timed(p.fn, 3) / 3
    per_call = max(min(t1, t3), 1e-6)
    calls = int(max(1, min(100000, round(a.target_sec / per_call))))
    ctx.tele.set_tag(test_id)
    mark = ctx.tele.mark()
    values = []
    t0 = time.time()
    for i in range(a.samples):
        dt = timed(p.fn, calls)
        v = p.work * calls / dt / p.scale
        values.append(v)
        ctx.log.debug("%s: sample %d/%d = %.4f (%d calls, %.3f s)", test_id, i + 1, a.samples, v, calls, dt)
    duration = time.time() - t0
    win = ctx.tele.window(mark)
    ctx.tele.set_tag("idle")
    return values, calls, win, duration


def record_result(ctx: Ctx, spec: TestSpec, status: str, values=None, calls=None, win=None,
                  duration=None, start_temp=None, batch=None, extra=None, error=None,
                  record_type="test_result"):
    stats = describe(values or [])
    win = win or {}
    perf_w = None
    if stats.get("median") and win.get("power_avg_w"):
        perf_w = stats["median"] / win["power_avg_w"]
    rec = {
        "type": record_type, "suite_version": SUITE_VERSION, "test_id": spec.id,
        "category": spec.category, "expectation": spec.expectation, "unit": spec.unit,
        "desc": spec.desc, "status": status, "batch": batch, "calls_per_sample": calls,
        "samples": values or [], "stats": stats, "telemetry": win,
        "perf_per_watt": perf_w, "start_temp_c": start_temp, "duration_s": duration,
        "extra": extra or {}, "error": error,
    }
    ctx.jsonl.write(rec)
    thr = win.get("throttle_share_pct") or {}
    ctx.csv_rows.append({
        "test_id": spec.id, "category": spec.category, "expectation": spec.expectation,
        "unit": spec.unit, "status": status, "n": stats.get("n", 0),
        "median": stats.get("median"), "mean": stats.get("mean"), "stdev": stats.get("stdev"),
        "cv_pct": stats.get("cv_pct"), "min": stats.get("min"), "max": stats.get("max"),
        "batch": batch, "calls_per_sample": calls,
        "power_avg_w": win.get("power_avg_w"), "power_max_w": win.get("power_max_w"),
        "power_limit_w": win.get("power_limit_w"), "perf_per_watt": perf_w,
        "sm_clock_avg_mhz": win.get("sm_clock_avg_mhz"), "temp_max_c": win.get("temp_max_c"),
        "start_temp_c": start_temp, "mem_used_max_mib": win.get("mem_used_max_mib"),
        "throttle": " ".join(f"{k}:{v}%" for k, v in thr.items()),
        "duration_s": duration, "error": error,
    })
    if status == "ok" and stats.get("n"):
        ctx.log.info(
            "%-28s %10.2f %-9s cv %4.1f%%  n=%d | %s W avg | %s MHz | %s °C max | %s",
            spec.id, stats["median"], spec.unit, stats["cv_pct"], stats["n"],
            _fmt(win.get("power_avg_w"), 1), _fmt(win.get("sm_clock_avg_mhz"), 0),
            _fmt(win.get("temp_max_c"), 0),
            ", ".join(f"{k} {v}%" for k, v in thr.items()) or "no throttle",
        )
    else:
        ctx.log.error("%-28s %s %s", spec.id, status.upper(), error or "")


def _fmt(v, nd):
    return "n/a" if v is None else f"{v:.{nd}f}"


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def run_throughput(ctx: Ctx, spec: TestSpec):
    start_temp = cooldown(ctx, spec.id)
    ctx.log.debug("%s: %s", spec.id, spec.desc)
    p = None
    used_batch = None
    last_err, last_oom = None, False
    for i, batch in enumerate(spec.batches):
        retry = False
        try:
            p = spec.prepare(batch)
            p.fn()          # validation call (fails fast on OOM / unsupported kernels)
            sync()
            used_batch = batch
        except Exception as e:  # noqa: BLE001
            # keep only text: the traceback would pin the model's GPU memory
            last_err, last_oom = repr(e)[:500], is_oom(e)
            p = None
            retry = last_oom and i < len(spec.batches) - 1
            if not last_oom:
                ctx.log.debug("%s: prepare failed", spec.id, exc_info=True)
        if p is not None:
            break
        free_gpu()
        if retry:
            ctx.log.warning("%s: out of memory at batch %s, retrying with batch %s",
                            spec.id, batch, spec.batches[i + 1])
            continue
        break
    if p is None:
        record_result(ctx, spec, "oom" if last_oom else "error", start_temp=start_temp, error=last_err)
        free_gpu()
        return
    try:
        values, calls, win, duration = measure(ctx, p, spec.id)
        extra = dict(p.extra)
        extra["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        record_result(ctx, spec, "ok", values, calls, win, duration, start_temp, used_batch, extra)
    except Exception as e:  # noqa: BLE001
        ctx.log.exception("%s failed during measurement", spec.id)
        record_result(ctx, spec, "oom" if is_oom(e) else "error", start_temp=start_temp,
                      batch=used_batch, error=repr(e)[:500])
    finally:
        del p
        free_gpu()


# =========================================================================== test builders

def make_gemm(ctx: Ctx, kind: str, n: int):
    def prepare(_batch):
        dev = ctx.dev
        g = torch.Generator(device=dev).manual_seed(n)
        if kind == "int8":
            a = torch.randint(-64, 64, (n, n), device=dev, dtype=torch.int8, generator=g)
            b = torch.randint(-64, 64, (n, n), device=dev, dtype=torch.int8, generator=g)
            fn = lambda: torch._int_mm(a, b)  # noqa: E731
        else:
            dt = torch.float32 if kind in ("tf32", "fp32") else DT[kind]
            a = torch.randn(n, n, device=dev, dtype=dt, generator=g)
            b = torch.randn(n, n, device=dev, dtype=dt, generator=g)
            c = torch.empty(n, n, device=dev, dtype=dt)
            tf32 = kind == "tf32"

            def fn():
                set_tf32(tf32)
                torch.mm(a, b, out=c)
        return Prepared(fn, 2.0 * n ** 3, 1e12, {"n": n, "dtype": kind,
                        "accumulate": "int32" if kind == "int8" else "fp32"})
    return prepare


def make_membw(ctx: Ctx, mib: int):
    def prepare(_batch):
        n = mib * 2**20 // 4
        a = torch.randn(n, device=ctx.dev)
        b = torch.empty_like(a)
        return Prepared(lambda: b.copy_(a), 2.0 * n * 4, 1e9, {"buffer_mib": mib})
    return prepare


def make_sdpa(ctx: Ctx, dtype: str, backward: bool, B=8, H=16, S=2048, D=64):
    def prepare(_batch):
        dt = DT[dtype]
        q, k, v = (torch.randn(B, H, S, D, device=ctx.dev, dtype=dt, requires_grad=backward)
                   for _ in range(3))
        flops_fwd = 4.0 * B * H * S * S * D
        if not backward:
            def fn():
                with torch.inference_mode():
                    F.scaled_dot_product_attention(q, k, v)
            return Prepared(fn, flops_fwd, 1e12, {"B": B, "H": H, "S": S, "D": D})
        go = torch.randn(B, H, S, D, device=ctx.dev, dtype=dt)

        def fn():
            q.grad = k.grad = v.grad = None
            F.scaled_dot_product_attention(q, k, v).backward(go)
        return Prepared(fn, 3.5 * flops_fwd, 1e12, {"B": B, "H": H, "S": S, "D": D,
                        "flop_model": "fwd + 2.5x fwd for bwd"})
    return prepare


def make_conv(ctx: Ctx, dtype: str, N=32, C=256, HW=56):
    def prepare(_batch):
        dt = DT[dtype]
        conv = nn.Conv2d(C, C, 3, padding=1, bias=False).to(ctx.dev, dt).to(memory_format=torch.channels_last)
        x = torch.randn(N, C, HW, HW, device=ctx.dev, dtype=dt).to(memory_format=torch.channels_last)

        def fn():
            with torch.inference_mode():
                conv(x)
        return Prepared(fn, 2.0 * N * HW * HW * C * C * 9, 1e12,
                        {"N": N, "C": C, "HW": HW, "kernel": "3x3", "layout": "channels_last"})
    return prepare


def make_llm_prefill(ctx: Ctx, prompt: int):
    def prepare(_batch):
        m = build_decoder_lowp(LLM_CFG, ctx.dev, torch.bfloat16)
        idx = torch.randint(0, LLM_CFG["vocab"], (1, prompt), device=ctx.dev)
        mp = m.matmul_params()

        def fn():
            with torch.inference_mode():
                m(idx, last_only=True)
        return Prepared(fn, float(prompt), 1.0, {
            "model": "synthetic Llama-style decoder, random weights", **LLM_CFG,
            "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
            "prompt_tokens": prompt, "batch": 1, "approx_flop_per_token": 2 * mp,
        })
    return prepare


def make_llm_decode(ctx: Ctx, batch_size: int, prompt=512, steps=128):
    def prepare(_batch):
        cfg = LLM_CFG
        m = build_decoder_lowp(cfg, ctx.dev, torch.bfloat16)
        H, D = cfg["heads"], cfg["dim"] // cfg["heads"]
        T = prompt + steps
        idx = torch.randint(0, cfg["vocab"], (batch_size, prompt), device=ctx.dev)
        with torch.inference_mode():
            caches = [(torch.zeros(batch_size, H, T, D, device=ctx.dev, dtype=torch.bfloat16),
                       torch.zeros(batch_size, H, T, D, device=ctx.dev, dtype=torch.bfloat16))
                      for _ in range(cfg["layers"])]
            first = m(idx, caches, 0, last_only=True).argmax(-1)

        def fn():
            with torch.inference_mode():
                tok = first
                for i in range(steps):
                    tok = m(tok, caches, prompt + i, last_only=True).argmax(-1)
        return Prepared(fn, float(batch_size * steps), 1.0, {
            "model": "synthetic Llama-style decoder, random weights", **cfg,
            "batch": batch_size, "prompt_tokens": prompt, "generated_tokens": steps,
            "kv_cache": "bf16, preallocated", "decoding": "greedy argmax, eager PyTorch",
        })
    return prepare


def make_gpt_train(ctx: Ctx, dtype: str, seq=1024):
    def prepare(batch):
        torch.manual_seed(0)
        with torch.device(ctx.dev):
            m = Decoder(**GPT_CFG, tie=True)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        dt = DT[dtype]
        scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
        x = torch.randint(0, GPT_CFG["vocab"], (batch, seq), device=ctx.dev)
        y = torch.randint(0, GPT_CFG["vocab"], (batch, seq), device=ctx.dev)

        def fn():
            with torch.autocast("cuda", dtype=dt):
                logits = m(x)
                loss = F.cross_entropy(logits.view(-1, GPT_CFG["vocab"]), y.view(-1))  # runs in FP32 under autocast
            del logits
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        return Prepared(fn, float(batch * seq), 1.0, {
            "model": "GPT-style decoder ~124M, tied embeddings (SwiGLU, RMSNorm)", **GPT_CFG, "seq": seq,
            "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
            "batch": batch, "optimizer": "AdamW fused", "precision": f"autocast {dtype}, FP32 master weights",
        })
    return prepare


def make_resnet(ctx: Ctx, train: bool):
    def prepare(batch):
        import torchvision
        torch.manual_seed(0)
        m = torchvision.models.resnet50(weights=None).to(ctx.dev).to(memory_format=torch.channels_last)
        if train:
            m.train()
            opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
            x = torch.randn(batch, 3, 224, 224, device=ctx.dev).to(memory_format=torch.channels_last)
            y = torch.randint(0, 1000, (batch,), device=ctx.dev)

            def fn():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(m(x), y)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            prec = "autocast bf16, FP32 weights, SGD"
        else:
            m = m.eval().half()
            x = torch.randn(batch, 3, 224, 224, device=ctx.dev, dtype=torch.float16).to(
                memory_format=torch.channels_last)

            def fn():
                with torch.inference_mode():
                    m(x)
            prec = "fp16 weights + activations"
        return Prepared(fn, float(batch), 1.0, {"model": "torchvision resnet50, random init",
                                                 "batch": batch, "res": 224, "precision": prec})
    return prepare


def make_sd(ctx: Ctx, model_id: str, steps=20):
    def prepare(_batch):
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(ctx.dev)
        pipe.set_progress_bar_config(disable=True)

        def fn():
            pipe("a red bicycle leaning against a brick wall, photo", num_inference_steps=steps,
                 guidance_scale=0.0, height=512, width=512,
                 generator=torch.Generator(device=ctx.dev).manual_seed(0))
        return Prepared(fn, float(steps), 1.0, {"model": model_id, "steps": steps, "res": 512,
                                                 "cfg": "off (guidance 0)", "dtype": "bf16",
                                                 "metric": "denoising steps/s incl. text encoder + VAE"})
    return prepare


# --------------------------------------------------------------------------- correctness

def run_correctness(ctx: Ctx, spec: TestSpec, kind: str, n=2048):
    """Deterministic GEMM: output hash (bit-exactness across phases) + error vs FP64."""
    try:
        dev = ctx.dev
        g = torch.Generator(device="cpu").manual_seed(1234)
        a32 = torch.randn(n, n, generator=g).to(dev)
        b32 = torch.randn(n, n, generator=g).to(dev)
        dt = torch.float32 if kind in ("tf32", "fp32") else DT[kind]
        a, b = a32.to(dt), b32.to(dt)
        set_tf32(kind == "tf32")
        hashes = []
        for _ in range(2):
            c = torch.mm(a, b)
            sync()
            raw = c.contiguous().view(torch.uint8 if c.element_size() == 1 else
                                      {2: torch.int16, 4: torch.int32}[c.element_size()]).cpu()
            try:
                blob = raw.numpy().tobytes()
            except Exception:  # noqa: BLE001
                blob = bytes(raw.view(torch.uint8).tolist())
            import hashlib
            hashes.append(hashlib.sha256(blob).hexdigest())
        ref = a.double() @ b.double()
        err = ((c.double() - ref).abs().max() / ref.abs().max()).item()
        mean_err = ((c.double() - ref).abs().mean() / ref.abs().mean()).item()
        set_tf32(False)
        # verify that the TF32 switch really works, otherwise the FP32 control test is meaningless
        tf32_ok = None
        if kind == "fp32":
            tf32_ok = err < 1e-4
            if not tf32_ok:
                ctx.log.error("FP32 GEMM rel. error %.2e: TF32 apparently NOT disabled – FP32 control invalid", err)
        if kind == "tf32":
            tf32_ok = err > 1e-5
            if not tf32_ok:
                ctx.log.error("TF32 GEMM rel. error %.2e: TF32 apparently NOT active – TF32 tests invalid", err)
        extra = {"n": n, "precision_switch_ok": tf32_ok, "dtype": kind, "sha256": hashes[0], "deterministic_within_run": hashes[0] == hashes[1],
                 "max_rel_err_vs_fp64": err, "mean_rel_err_vs_fp64": mean_err}
        ctx.log.info("%-28s sha256 %s… | deterministic=%s | max rel err %.3e | mean rel err %.3e",
                     spec.id, hashes[0][:16], hashes[0] == hashes[1], err, mean_err)
        ctx.jsonl.write({"type": "correctness", "suite_version": SUITE_VERSION, "test_id": spec.id,
                         "category": spec.category, "expectation": spec.expectation, "status": "ok", **extra})
    except Exception as e:  # noqa: BLE001
        ctx.log.exception("%s failed", spec.id)
        ctx.jsonl.write({"type": "correctness", "test_id": spec.id, "status": "error", "error": repr(e)[:500]})
    finally:
        free_gpu()


# --------------------------------------------------------------------------- sustained

def run_sustained(ctx: Ctx, spec: TestSpec, kind: str, n: int, seconds: float, window_s=5.0):
    start_temp = cooldown(ctx, spec.id)
    p = None
    try:
        p = make_gemm(ctx, kind, n)(None)
        for _ in range(3):
            p.fn()
        sync()
        per_call = timed(p.fn, 3) / 3
        calls = max(1, round(window_s / per_call))
        path = os.path.join(ctx.run_dir, f"sustained_{spec.id}.csv")
        fields = ["window", "t_s", "value", "power_avg_w", "power_max_w", "sm_clock_avg_mhz",
                  "temp_avg_c", "temp_max_c", "throttle"]
        ctx.tele.set_tag(spec.id)
        mark0 = ctx.tele.mark()
        values, rows = [], []
        t0 = time.time()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            i = 0
            while time.time() - t0 < seconds:
                mk = ctx.tele.mark()
                dt = timed(p.fn, calls)
                v = p.work * calls / dt / p.scale
                win = ctx.tele.window(mk)
                row = {"window": i, "t_s": round(time.time() - t0, 2), "value": round(v, 4),
                       "power_avg_w": _r(win.get("power_avg_w")), "power_max_w": _r(win.get("power_max_w")),
                       "sm_clock_avg_mhz": _r(win.get("sm_clock_avg_mhz")),
                       "temp_avg_c": _r(win.get("temp_avg_c")), "temp_max_c": win.get("temp_max_c"),
                       "throttle": " ".join(f"{k}:{v2}%" for k, v2 in (win.get("throttle_share_pct") or {}).items())}
                w.writerow(row)
                fh.flush()
                rows.append(row)
                values.append(v)
                ctx.log.debug("%s window %d: %.2f %s %s W %s MHz %s °C", spec.id, i, v, spec.unit,
                              row["power_avg_w"], row["sm_clock_avg_mhz"], row["temp_max_c"])
                i += 1
        duration = time.time() - t0
        win_all = ctx.tele.window(mark0)
        ctx.tele.set_tag("idle")
        steady = [r["value"] for r in rows if r["t_s"] > 30.0] or values
        extra = {"n": n, "dtype": kind, "window_s": window_s, "seconds": seconds,
                 "first_window": values[0] if values else None,
                 "steady_median_after_30s": describe(steady).get("median"),
                 "last_60s_median": describe([r["value"] for r in rows if r["t_s"] > duration - 60]).get("median"),
                 "timeline_csv": os.path.basename(path)}
        record_result(ctx, spec, "ok", values, calls, win_all, duration, start_temp, None, extra)
    except Exception as e:  # noqa: BLE001
        ctx.log.exception("%s failed", spec.id)
        record_result(ctx, spec, "oom" if is_oom(e) else "error", start_temp=start_temp, error=repr(e)[:500])
    finally:
        del p
        free_gpu()


def _r(v, nd=2):
    return None if v is None else round(v, nd)


# =========================================================================== catalogue

def catalogue(ctx: Ctx) -> list[TestSpec]:
    a = ctx.args
    sizes = [2048] if a.smoke else [2048, 4096, 8192]
    T: list[TestSpec] = []

    # 1) correctness first (cheap, no cooldown needed)
    for k in ("fp16", "bf16", "tf32", "fp32"):
        T.append(TestSpec(f"correct_gemm_{k}", "correctness", INFO, "-",
                          f"{k} 2048³ GEMM: SHA-256 of output and relative error vs FP64", kind="correctness"))

    # 2) GEMM micro-benchmarks
    gemm_desc = {
        "fp16": ("TFLOPS", GAIN, "FP16 inputs, FP32 accumulate (cuBLAS)"),
        "bf16": ("TFLOPS", GAIN, "BF16 inputs, FP32 accumulate (cuBLAS)"),
        "tf32": ("TFLOPS", GAIN, "FP32 inputs via TF32 tensor cores, FP32 accumulate"),
        "fp32": ("TFLOPS", CONTROL, "FP32 on CUDA cores, TF32 disabled (control)"),
        "int8": ("TOPS", CONTROL, "INT8 inputs, INT32 accumulate via torch._int_mm (control)"),
    }
    for n in sizes:
        for k, (unit, exp, d) in gemm_desc.items():
            T.append(TestSpec(f"gemm_{k}_{n}", "gemm", exp, unit, f"Dense {n}³ GEMM, {d}",
                              make_gemm(ctx, k, n)))
    T.append(TestSpec("membw_copy_1gib", "memory", CONTROL, "GB/s",
                      "Device-to-device copy of 1 GiB, read+write bytes (control)", make_membw(ctx, 1024)))

    # 3) kernels
    T.append(TestSpec("sdpa_fwd_fp16", "kernel", GAIN, "TFLOPS",
                      "scaled_dot_product_attention forward, B8 H16 S2048 D64, FP16", make_sdpa(ctx, "fp16", False)))
    T.append(TestSpec("sdpa_fwd_bf16", "kernel", GAIN, "TFLOPS",
                      "scaled_dot_product_attention forward, B8 H16 S2048 D64, BF16", make_sdpa(ctx, "bf16", False)))
    T.append(TestSpec("sdpa_fwdbwd_bf16", "kernel", GAIN, "TFLOPS",
                      "scaled_dot_product_attention forward+backward, BF16", make_sdpa(ctx, "bf16", True)))
    T.append(TestSpec("conv3x3_fp16", "kernel", PARTIAL, "TFLOPS",
                      "Conv2d 3x3 256->256 @56x56 N32 channels_last, FP16 (cuDNN)", make_conv(ctx, "fp16")))
    T.append(TestSpec("conv3x3_bf16", "kernel", PARTIAL, "TFLOPS",
                      "Conv2d 3x3 256->256 @56x56 N32 channels_last, BF16 (cuDNN)", make_conv(ctx, "bf16")))

    # 4) end-to-end workloads
    # Fixed batch via CLI makes both phases use the same size (no OOM fallback differences).
    gpt_b = [a.gpt_batch] if a.gpt_batch else [4, 2, 1]
    rn_b = [a.resnet_batch] if a.resnet_batch else [64, 32, 16]
    prompts = [512] if a.smoke else [512, 2048]
    for pr in prompts:
        T.append(TestSpec(f"llm_prefill_bf16_p{pr}", "workload", GAIN, "tokens/s",
                          f"LLM prompt processing, ~1B Llama-style model, BF16, {pr}-token prompt",
                          make_llm_prefill(ctx, pr)))
    steps = 16 if a.smoke else 128
    T.append(TestSpec("llm_decode_bf16_b1", "workload", INFO, "tokens/s",
                      "LLM token generation batch 1 (bandwidth/launch-overhead bound, eager PyTorch; "
                      "expected ~unchanged, informational)",
                      make_llm_decode(ctx, 1, steps=steps)))
    T.append(TestSpec("llm_decode_bf16_b16", "workload", PARTIAL, "tokens/s",
                      "LLM token generation batch 16 (more compute-bound)", make_llm_decode(ctx, 16, steps=steps)))
    T.append(TestSpec("train_gpt124m_bf16", "workload", GAIN, "tokens/s",
                      "Training step GPT-style 124M, seq 1024, autocast BF16, fused AdamW",
                      make_gpt_train(ctx, "bf16"), batches=gpt_b))
    T.append(TestSpec("train_gpt124m_fp16", "workload", GAIN, "tokens/s",
                      "Training step GPT-style 124M, seq 1024, autocast FP16 + GradScaler",
                      make_gpt_train(ctx, "fp16"), batches=gpt_b))
    if a.torchvision:
        T.append(TestSpec("resnet50_train_bf16", "workload", PARTIAL, "images/s",
                          "ResNet-50 training 224px, autocast BF16, channels_last",
                          make_resnet(ctx, True), batches=rn_b))
        T.append(TestSpec("resnet50_infer_fp16", "workload", PARTIAL, "images/s",
                          "ResNet-50 inference 224px, FP16, channels_last",
                          make_resnet(ctx, False), batches=[128, 64, 32]))
    if a.sd_model:
        T.append(TestSpec("sd_pipeline_bf16_512", "workload", GAIN, "it/s",
                          f"Diffusers text-to-image {a.sd_model}, 512x512, 20 steps, BF16",
                          make_sd(ctx, a.sd_model)))

    # 5) sustained load
    T.append(TestSpec("sustained_gemm_bf16_8192", "sustained", GAIN, "TFLOPS",
                      f"BF16 8192³ GEMM continuously for {a.sustained_sec:.0f} s, 5 s windows "
                      "(power/thermal behaviour)", kind="sustained"))
    return T


# =========================================================================== main

def phase_check(env: dict, args, log) -> None:
    state = env["gpu_unlock"]["state"]
    log.info("gpu_unlock state for this boot: %s (cmdline targets=%s, module loaded=%s, log lines=%d)",
             state, env["gpu_unlock"]["cmdline_targets"], env["gpu_unlock"]["module_loaded"],
             env["gpu_unlock"]["log_lines"])
    problems = []
    if state == "unknown":
        groups = set(sh("id -nG")[1].split())
        if not ({"adm", "systemd-journal"} & groups):
            problems.append("kernel log not readable for this user: run `sudo usermod -aG adm $USER`, "
                            "then log out and in again")
    if args.phase == "baseline" and state in ("unlocked", "unknown"):
        problems.append(f"phase 'baseline' but gpu_unlock state is '{state}' – remove gpu_unlock.targets= "
                        "from the kernel command line and reboot")
    if args.phase == "unlocked" and state != "unlocked":
        problems.append(f"phase 'unlocked' but gpu_unlock state is '{state}' – check "
                        "`journalctl -k -b | grep gpu_unlock`")
    if env.get("ac_online") is False:
        problems.append("running on battery – connect the power adapter")
    if env.get("platform_profile") and env["platform_profile"] != "performance":
        problems.append(f"platform_profile is '{env['platform_profile']}', expected 'performance' "
                        "(powerprofilesctl set performance)")
    if (env.get("nvidia_runtime_d3") or "").lower().startswith("enabled"):
        # Required in the unlocked phase (power-off resets the unlock registers) and therefore
        # also in the baseline, so both phases run under identical power management.
        problems.append("NVIDIA Runtime D3 is enabled: the GPU may power off when idle, which resets the "
                        "unlock registers. Run `./run.sh prepare` (NVreg_DynamicPowerManagement=0x00) and reboot")
    for p in problems:
        (log.warning if args.smoke else log.error)("PREFLIGHT: %s", p)
    if problems and args.smoke:
        log.warning("Smoke test: preflight problems are reported but do not stop the run")
    elif problems and not args.force:
        log.critical("Preflight failed (%d problem(s)). Fix them or re-run with --force "
                     "(the run will then be marked as forced).", len(problems))
        sys.exit(3)
    env["preflight_problems"] = problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["baseline", "unlocked", "auto"], required=True,
                    help="auto = derive from the gpu_unlock state of this boot")
    ap.add_argument("--out-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    ap.add_argument("--label", default="", help="free-text note stored with the run")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--target-sec", type=float, default=2.0, help="duration of one sample")
    ap.add_argument("--warmup-sec", type=float, default=3.0)
    ap.add_argument("--cooldown-temp", type=float, default=55.0, help="start each test below this GPU temp")
    ap.add_argument("--cooldown-min", type=float, default=10.0)
    ap.add_argument("--cooldown-max", type=float, default=120.0)
    ap.add_argument("--sustained-sec", type=float, default=180.0)
    ap.add_argument("--idle-sec", type=float, default=15.0, help="idle telemetry before the tests")
    ap.add_argument("--only", default=None, help="regex: run only matching test ids")
    ap.add_argument("--skip", default=None, help="regex: skip matching test ids")
    ap.add_argument("--sd-model", default=None, help="optional diffusers model id, e.g. stabilityai/sd-turbo")
    ap.add_argument("--no-torchvision", dest="torchvision", action="store_false")
    ap.add_argument("--gpt-batch", type=int, default=None, help="fixed batch for GPT training (default: 4, OOM fallback)")
    ap.add_argument("--resnet-batch", type=int, default=None, help="fixed batch for ResNet-50 training (default: 64, OOM fallback)")
    ap.add_argument("--gpu-unlock-dir", default=os.path.expanduser("~/gpu-unlock"))
    ap.add_argument("--smoke", action="store_true", help="2-3 minute functional check, results not comparable")
    ap.add_argument("--force", action="store_true", help="continue despite failed preflight checks")
    ap.add_argument("--list", action="store_true", help="list tests and exit")
    args = ap.parse_args()

    if args.smoke:
        args.samples, args.target_sec, args.warmup_sec = 2, 0.5, 0.5
        args.cooldown_min, args.cooldown_max, args.sustained_sec, args.idle_sec = 0.0, 0.0, 15.0, 2.0
    if args.torchvision:
        try:
            import torchvision  # noqa: F401
        except ImportError:
            args.torchvision = False

    if args.list:
        class _Dummy:
            pass
        d = _Dummy()
        d.args = args
        d.dev = None
        for t in catalogue(d):
            print(f"{t.id:28s} {t.category:12s} {t.expectation:8s} {t.unit:9s} {t.desc}")
        return

    if args.phase == "auto":
        args.phase = "unlocked" if sysinfo.quick_state()["state"] == "unlocked" else "baseline"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{args.phase}" + ("_smoke" if args.smoke else "") + (
        "_" + re.sub(r"[^A-Za-z0-9_-]+", "-", args.label) if args.label else "")
    run_dir = os.path.join(args.out_root, name)
    os.makedirs(run_dir, exist_ok=True)
    log = setup_logging(os.path.join(run_dir, "run.log"))
    log.info("=== RTX 30 gpu-unlock benchmark suite v%s | phase=%s%s ===", SUITE_VERSION, args.phase,
             " | SMOKE TEST" if args.smoke else "")
    log.info("Run directory: %s", run_dir)
    log.info("Arguments: %s", json.dumps(vars(args), default=str))

    if not torch.cuda.is_available():
        log.critical("CUDA is not available to PyTorch (torch %s). Is the NVIDIA driver loaded?", torch.__version__)
        sys.exit(2)

    env = sysinfo.capture(os.path.join(run_dir, "env"), args.gpu_unlock_dir)
    env["torch"] = {
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "python": sys.version.split()[0], "device": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "total_mem_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
    }
    try:
        import torchvision
        env["torch"]["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    env["torch"]["matmul_settings"] = {
        "fp16_reduced_precision_reduction": False, "bf16_reduced_precision_reduction": False,
        "cudnn_benchmark": True,
        "env_overrides": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                                                         "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG")},
    }
    env["run"] = {"suite_version": SUITE_VERSION, "phase": args.phase, "smoke": args.smoke,
                  "label": args.label, "started": now_iso(), "args": vars(args), "run_dir": name}
    g = env["gpu"]
    log.info("GPU: %s | PCI %s (%s) | VBIOS %s | driver %s (%s) | %s SMs | %s MiB",
             g.get("name"), g.get("pci.bus_id"), g.get("pci.device_id"), g.get("vbios_version"),
             g.get("driver_version"), env["driver_flavour"], env["torch"]["sm_count"], env["torch"]["total_mem_mib"])
    log.info("Power limit: current %s | default %s | max %s | Dynamic Boost %s | nvidia-powerd %s",
             g.get("power.limit"), g.get("power.default_limit"), g.get("power.max_limit"),
             env.get("notebook_dynamic_boost"), env.get("nvidia_powerd"))
    log.info("Host: %s | %s | kernel %s | CPU %s", env["host"], env["os"], env["kernel"], env["cpu"])
    log.info("PyTorch %s | CUDA %s | cuDNN %s | profile %s | AC %s | Runtime D3 %s",
             env["torch"]["torch"], env["torch"]["cuda"], env["torch"]["cudnn"], env.get("platform_profile"),
             env.get("ac_online"), env.get("nvidia_runtime_d3"))
    phase_check(env, args, log)
    env["run"]["forced"] = bool(env.get("preflight_problems"))
    with open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, default=str)

    # global, identical settings for both phases
    torch.backends.cudnn.benchmark = True
    set_tf32(False)
    # FP32 accumulation also for split-K reductions -> labels are true and hashes reproducible
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    try:
        max_w = float((g.get("power.max_limit") or "").split()[0])
    except (ValueError, IndexError):
        max_w = 0.0
    tele = Telemetry(os.path.join(run_dir, "telemetry.csv"),
                     power_sanity_max_w=max(250.0, 1.5 * max_w) if max_w else 400.0, log=log).start()
    log.info("Telemetry backend: %s (interval %.1f s)", tele.backend, tele.interval)
    jsonl = JsonlWriter(os.path.join(run_dir, "results.jsonl"))
    ctx = Ctx(args, log, tele, jsonl, run_dir)
    jsonl.write({"type": "run_start", "env": env})

    # idle reference
    tele.set_tag("idle_reference")
    mk = tele.mark()
    time.sleep(args.idle_sec)
    idle = tele.window(mk)
    tele.set_tag("idle")
    jsonl.write({"type": "idle_reference", "telemetry": idle})
    log.info("Idle reference (%ds): %s W, %s °C, %s MHz", args.idle_sec, _fmt(idle.get("power_avg_w"), 1),
             _fmt(idle.get("temp_avg_c"), 0), _fmt(idle.get("sm_clock_avg_mhz"), 0))

    tests = catalogue(ctx)
    if args.only:
        tests = [t for t in tests if re.search(args.only, t.id)]
    if args.skip:
        tests = [t for t in tests if not re.search(args.skip, t.id)]
    log.info("Running %d tests (%d samples × %.1f s each, cooldown to %.0f °C)", len(tests), args.samples,
             args.target_sec, args.cooldown_temp)
    t_start = time.time()
    for i, spec in enumerate(tests, 1):
        log.info("[%d/%d] %s", i, len(tests), spec.id)
        if spec.kind == "correctness":
            run_correctness(ctx, spec, spec.id.rsplit("_", 1)[1])
        elif spec.kind == "sustained":
            run_sustained(ctx, spec, "bf16", 2048 if args.smoke else 8192, args.sustained_sec)
        else:
            run_throughput(ctx, spec)
    total = time.time() - t_start

    jsonl.write({"type": "run_end", "duration_s": total, "finished": now_iso()})
    jsonl.close()
    tele.stop()

    fields = list(ctx.csv_rows[0].keys()) if ctx.csv_rows else []
    with open(os.path.join(run_dir, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in ctx.csv_rows:
            w.writerow(r)
    write_summary(run_dir, env, ctx.csv_rows, total)
    write_manifest(run_dir)
    n_bad = sum(1 for r in ctx.csv_rows if r["status"] != "ok")
    log.info("=== Finished in %.1f min: %d tests ok, %d failed. Results: %s ===", total / 60,
             len(ctx.csv_rows) - n_bad, n_bad, run_dir)


def write_summary(run_dir, env, rows, total):
    g = env["gpu"]
    lines = [
        f"# Run summary – {env['run']['run_dir']}", "",
        f"- Phase: **{env['run']['phase']}**{' (smoke test)' if env['run']['smoke'] else ''}"
        f"{' (FORCED despite preflight problems)' if env['run'].get('forced') else ''}",
        f"- gpu_unlock state: **{env['gpu_unlock']['state']}**",
        f"- GPU: {g.get('name')} ({g.get('pci.device_id')}), VBIOS {g.get('vbios_version')}, "
        f"driver {g.get('driver_version')} ({env['driver_flavour']})",
        f"- Power limit {g.get('power.limit')} (max {g.get('power.max_limit')}), profile "
        f"{env.get('platform_profile')}, AC {env.get('ac_online')}, Runtime D3 {env.get('nvidia_runtime_d3')}",
        f"- PyTorch {env['torch']['torch']}, CUDA {env['torch']['cuda']}, cuDNN {env['torch']['cudnn']}",
        f"- Duration {total / 60:.1f} min", "",
        "| Test | Category | Expect | Median | Unit | CV % | n | Avg W | Max °C | Status |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append("| {test_id} | {category} | {expectation} | {m} | {unit} | {cv} | {n} | {w} | {t} | {status} |".format(
            m=_fmt(r["median"], 2), cv=_fmt(r["cv_pct"], 1), w=_fmt(r["power_avg_w"], 1),
            t=_fmt(r["temp_max_c"], 0), **{k: r[k] for k in ("test_id", "category", "expectation", "unit", "n", "status")}))
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_manifest(run_dir):
    files = {}
    for root, _, names in os.walk(run_dir):
        for n in sorted(names):
            if n == "manifest.json":
                continue
            p = os.path.join(root, n)
            files[os.path.relpath(p, run_dir)] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"generated": now_iso(), "files": files}, fh, indent=2)


if __name__ == "__main__":
    main()
