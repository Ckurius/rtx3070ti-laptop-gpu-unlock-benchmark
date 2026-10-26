#!/usr/bin/env python3
"""Compare a baseline run with an unlocked run and write a publishable report.

  python3 bench/compare.py                       # newest non-smoke baseline vs newest unlocked
  python3 bench/compare.py --baseline results/A --unlocked results/B

Output: reports/<timestamp>_<baseline>_vs_<unlocked>/
  report.md          GitHub-ready report (methodology, validity checks, all results, charts)
  compat_report.md   pre-filled text for the gpu-unlock "Compatibility report" issue
  comparison.csv     one row per test with both phases and the change
  comparison.json    same data plus validity checks, machine-readable
  compare.log        log of this comparison
  charts/*.png       figures referenced by report.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SUITE_VERSION, describe, now_iso, read_jsonl, setup_logging  # noqa: E402
import charts as ch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT_ORDER = ["gemm", "memory", "kernel", "workload", "sustained"]
CAT_TITLE = {"gemm": "GEMM micro-benchmarks", "memory": "Memory bandwidth", "kernel": "Kernels",
             "workload": "End-to-end workloads", "sustained": "Sustained load"}
HEADLINE = [
    ("gemm_bf16_8192", "BF16 GEMM 8192³ (FP32 acc.)"),
    ("gemm_fp16_8192", "FP16 GEMM 8192³ (FP32 acc.)"),
    ("gemm_tf32_8192", "TF32 GEMM 8192³"),
    ("sdpa_fwdbwd_bf16", "Attention fwd+bwd, BF16"),
    ("llm_prefill_bf16_p2048", "LLM prompt processing (2048 tok)"),
    ("llm_decode_bf16_b1", "LLM token generation (batch 1)"),
    ("train_gpt124m_bf16", "GPT-124M training, BF16"),
    ("resnet50_train_bf16", "ResNet-50 training, BF16"),
    ("sd_pipeline_bf16_512", "Stable Diffusion 512², BF16"),
    ("sustained_gemm_bf16_8192", "Sustained BF16 GEMM (median of all windows)"),
]


# --------------------------------------------------------------------------- loading

def find_runs(phase: str) -> list[str]:
    runs = sorted(d for d in glob.glob(os.path.join(ROOT, "results", f"*_{phase}*"))
                  if os.path.isdir(d) and "_smoke" not in os.path.basename(d)
                  and os.path.exists(os.path.join(d, "results.jsonl")))
    return runs


def load_run(path: str) -> dict:
    recs = read_jsonl(os.path.join(path, "results.jsonl"))
    env = json.load(open(os.path.join(path, "env.json"), encoding="utf-8"))
    tests = {r["test_id"]: r for r in recs if r.get("type") == "test_result"}
    corr = {r["test_id"]: r for r in recs if r.get("type") == "correctness"}
    idle = next((r for r in recs if r.get("type") == "idle_reference"), {})
    end = next((r for r in recs if r.get("type") == "run_end"), {})
    return {"path": path, "name": os.path.basename(path), "env": env, "tests": tests,
            "correctness": corr, "idle": idle.get("telemetry", {}), "end": end}


def load_timeline(run: dict, test_id: str) -> list[dict]:
    p = os.path.join(run["path"], f"sustained_{test_id}.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- analysis

def compare_test(b: dict | None, u: dict | None) -> dict:
    ref = u or b
    row = {"test_id": ref["test_id"], "category": ref["category"], "expectation": ref["expectation"],
           "unit": ref["unit"], "desc": ref.get("desc", ""),
           "status_b": b["status"] if b else "missing", "status_u": u["status"] if u else "missing"}
    if not (b and u and b["status"] == "ok" and u["status"] == "ok"):
        return row
    sb, su = b["stats"], u["stats"]
    tb, tu = b.get("telemetry") or {}, u.get("telemetry") or {}
    speed = su["median"] / sb["median"]
    lo, hi = su["min"] / sb["max"], su["max"] / sb["min"]
    row.update({
        "median_b": sb["median"], "median_u": su["median"], "min_b": sb["min"], "max_b": sb["max"],
        "min_u": su["min"], "max_u": su["max"], "cv_b": sb["cv_pct"], "cv_u": su["cv_pct"],
        "n_b": sb["n"], "n_u": su["n"], "speedup": speed, "change_pct": (speed - 1) * 100,
        "range_lo_pct": (lo - 1) * 100, "range_hi_pct": (hi - 1) * 100,
        # complete separation of the two sample sets <=> exact Mann-Whitney p = 2/C(n_b+n_u, n_b)
        "separated": su["min"] > sb["max"] or su["max"] < sb["min"],
        "p_exact": _mw_separated_p(sb["n"], su["n"]) if (su["min"] > sb["max"] or su["max"] < sb["min"]) else None,
        "power_b": tb.get("power_avg_w"), "power_u": tu.get("power_avg_w"),
        "clock_b": tb.get("sm_clock_avg_mhz"), "clock_u": tu.get("sm_clock_avg_mhz"),
        "temp_b": tb.get("temp_max_c"), "temp_u": tu.get("temp_max_c"),
        "start_temp_b": b.get("start_temp_c"), "start_temp_u": u.get("start_temp_c"),
        "ppw_b": b.get("perf_per_watt"), "ppw_u": u.get("perf_per_watt"),
        "batch_b": b.get("batch"), "batch_u": u.get("batch"),
        "throttle_b": tb.get("throttle_share_pct") or {}, "throttle_u": tu.get("throttle_share_pct") or {},
    })
    if row["ppw_b"] and row["ppw_u"]:
        row["ppw_change_pct"] = (row["ppw_u"] / row["ppw_b"] - 1) * 100
    return row


def _mw_separated_p(n1: int, n2: int) -> float:
    from math import comb
    return 2.0 / comb(n1 + n2, n1)


def validity(base: dict, unl: dict, rows: list[dict], tol: float) -> list[dict]:
    eb, eu = base["env"], unl["env"]
    checks = []

    def add(name, ok, detail, severity="error"):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "severity": severity})

    add("Baseline boot is stock", eb["gpu_unlock"]["state"] == "stock",
        f"state={eb['gpu_unlock']['state']}, cmdline targets={eb['gpu_unlock']['cmdline_targets']}")
    add("Unlocked boot reports 'gpu_unlock: unlocked'", eu["gpu_unlock"]["state"] == "unlocked",
        f"state={eu['gpu_unlock']['state']}, log lines={eu['gpu_unlock']['log_lines']}")
    for key, label in [("driver_version", "NVIDIA driver"), ("vbios_version", "VBIOS"),
                       ("power.max_limit", "Max power limit"), ("pci.device_id", "PCI device")]:
        vb, vu = eb["gpu"].get(key), eu["gpu"].get(key)
        add(f"Same {label}", vb == vu, f"{vb} / {vu}")
    for key, label in [("kernel", "kernel"), ("platform_profile", "platform profile"), ("ac_online", "AC power"),
                       ("nvidia_runtime_d3", "Runtime D3 setting")]:
        add(f"Same {label}", eb.get(key) == eu.get(key), f"{eb.get(key)} / {eu.get(key)}",
            "warning" if key == "nvidia_runtime_d3" else "error")
    add("Same PyTorch / CUDA / cuDNN", (eb["torch"]["torch"], eb["torch"]["cuda"], eb["torch"]["cudnn"]) ==
        (eu["torch"]["torch"], eu["torch"]["cuda"], eu["torch"]["cudnn"]),
        f"{eb['torch']['torch']}/{eb['torch']['cuda']}/{eb['torch']['cudnn']} vs "
        f"{eu['torch']['torch']}/{eu['torch']['cuda']}/{eu['torch']['cudnn']}")
    add("Same suite version", eb["run"]["suite_version"] == eu["run"]["suite_version"],
        f"{eb['run']['suite_version']} / {eu['run']['suite_version']}")
    add("No run was forced past preflight", not (eb["run"].get("forced") or eu["run"].get("forced")),
        f"{eb.get('preflight_problems') or '-'} / {eu.get('preflight_problems') or '-'}", "warning")
    ctrl = [r for r in rows if r["expectation"] == "control" and "change_pct" in r]
    bad = [r for r in ctrl if abs(r["change_pct"]) > tol]
    add(f"Control tests unchanged (±{tol:g} %)", ctrl and not bad,
        ", ".join(f"{r['test_id']} {r['change_pct']:+.1f}%" for r in ctrl) or "no control results")
    for r in rows:
        if "batch_b" in r and r["batch_b"] != r["batch_u"]:
            add(f"Same batch size for {r['test_id']}", False,
                f"{r['batch_b']} / {r['batch_u']} – excluded from comparison; re-run with --gpt-batch/--resnet-batch",
                "error")
            for k in ("change_pct", "speedup", "separated", "p_exact", "range_lo_pct", "range_hi_pct"):
                r.pop(k, None)
    for phase, run in (("baseline", base), ("unlocked", unl)):
        for tid, cr in run["correctness"].items():
            if cr.get("precision_switch_ok") is False:
                add(f"TF32 switch effective ({phase}, {tid})", False,
                    f"max rel err {cr.get('max_rel_err_vs_fp64')}", "error")
    cvs = [max(r["cv_b"], r["cv_u"]) for r in rows if "cv_b" in r]
    if cvs:
        worst = max(cvs)
        add("Measurement noise acceptable (max CV < 5 %)", worst < 5.0,
            f"median CV {describe(cvs)['median']:.2f} %, worst {worst:.2f} %", "warning")
    return checks


def correctness_rows(base, unl):
    out = []
    for tid in sorted(set(base["correctness"]) | set(unl["correctness"])):
        b, u = base["correctness"].get(tid, {}), unl["correctness"].get(tid, {})
        out.append({"test_id": tid, "dtype": (b or u).get("dtype"),
                    "identical": bool(b.get("sha256")) and b.get("sha256") == u.get("sha256"),
                    "sha_b": (b.get("sha256") or "")[:16], "sha_u": (u.get("sha256") or "")[:16],
                    "err_b": b.get("max_rel_err_vs_fp64"), "err_u": u.get("max_rel_err_vs_fp64"),
                    "det_b": b.get("deterministic_within_run"), "det_u": u.get("deterministic_within_run")})
    return out


# --------------------------------------------------------------------------- charts

def make_charts(rows, base, unl, out_dir, log) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed – charts skipped")
        return {}
    os.makedirs(out_dir, exist_ok=True)
    charts = {}
    C_B, C_U, C_CTRL, C_GAIN = "#8a8f98", "#2f6fdb", "#b8bcc4", "#2f6fdb"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})

    ok = [r for r in rows if "change_pct" in r]
    # 1) overview of all changes
    if ok:
        ordered = sorted(ok, key=lambda r: (CAT_ORDER.index(r["category"]) if r["category"] in CAT_ORDER else 9,
                                            r["test_id"]))
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(ordered) + 1.4))
        y = range(len(ordered))
        vals = [r["change_pct"] for r in ordered]
        cols = [C_CTRL if r["expectation"] == "control" else C_GAIN for r in ordered]
        ax.barh(list(y), vals, color=cols)
        for i, r in enumerate(ordered):
            ax.plot([r["range_lo_pct"], r["range_hi_pct"]], [i, i], color="#1b1f24", lw=1)
            ax.text(max(vals[i], r["range_hi_pct"]) + 1.5, i, f"{vals[i]:+.1f}%", va="center", fontsize=8)
        ax.set_yticks(list(y), [r["test_id"] for r in ordered], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="#1b1f24", lw=0.8)
        ax.set_xlabel("Change unlocked vs. baseline (median, %). Line = min/max range of all samples")
        ax.set_title("gpu-unlock effect per test (grey = control tests, expected 0 %)")
        fig.tight_layout()
        p = os.path.join(out_dir, "overview_change.png")
        fig.savefig(p)
        plt.close(fig)
        charts["overview"] = p

    # 2) GEMM throughput vs size
    gem = [r for r in ok if r["category"] == "gemm"]
    if gem:
        kinds = ["bf16", "fp16", "tf32", "fp32", "int8"]
        fig, axes = plt.subplots(1, len(kinds), figsize=(13, 3.2), sharey=False)
        for ax, k in zip(axes, kinds):
            rr = sorted((r for r in gem if r["test_id"].startswith(f"gemm_{k}_")),
                        key=lambda r: int(r["test_id"].rsplit("_", 1)[1]))
            if not rr:
                ax.set_visible(False)
                continue
            xs = [r["test_id"].rsplit("_", 1)[1] for r in rr]
            xi = range(len(xs))
            ax.bar([i - 0.2 for i in xi], [r["median_b"] for r in rr], 0.4, color=C_B, label="baseline")
            ax.bar([i + 0.2 for i in xi], [r["median_u"] for r in rr], 0.4, color=C_U, label="unlocked")
            ax.set_xticks(list(xi), xs)
            ax.set_title(f"{k.upper()} ({rr[0]['unit']})")
            ax.set_xlabel("N (N×N×N)")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Dense GEMM throughput, median of samples")
        fig.tight_layout()
        p = os.path.join(out_dir, "gemm_throughput.png")
        fig.savefig(p)
        plt.close(fig)
        charts["gemm"] = p

    # 3) sustained timeline
    tb, tu = load_timeline(base, "sustained_gemm_bf16_8192"), load_timeline(unl, "sustained_gemm_bf16_8192")
    if tb and tu:
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for tl, col, lab in ((tb, C_B, "baseline"), (tu, C_U, "unlocked")):
            t = [float(r["t_s"]) for r in tl]
            axes[0].plot(t, [float(r["value"]) for r in tl], color=col, label=lab)
            axes[1].plot(t, [_f(r["power_avg_w"]) for r in tl], color=col)
            axes[2].plot(t, [_f(r["sm_clock_avg_mhz"]) for r in tl], color=col)
            ax2 = axes[2]
        axes[0].set_ylabel("TFLOPS")
        axes[1].set_ylabel("GPU power (W)")
        ax2.set_ylabel("SM clock (MHz)")
        ax2.set_xlabel("time (s)")
        axes[0].legend(frameon=False)
        axes[0].set_title("Sustained BF16 8192³ GEMM – throughput, power and clock (5 s windows)")
        fig.tight_layout()
        p = os.path.join(out_dir, "sustained_bf16.png")
        fig.savefig(p)
        plt.close(fig)
        charts["sustained"] = p
    return charts


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- report

def fmt(v, nd=2, unit=""):
    if v is None:
        return "–"
    if isinstance(v, float) and v != v:
        return "–"
    return f"{v:,.{nd}f}{unit}"


def sci(v):
    return "–" if v is None else f"{v:.3e}"


def pct(v, nd=1):
    return "–" if v is None else f"{v:+.{nd}f} %"


def write_report(out, base, unl, rows, checks, corr, charts, rel_results, system=None, repeats=None):
    eb, eu = base["env"], unl["env"]
    g = eu["gpu"]
    rel = lambda p: os.path.relpath(p, out)  # noqa: E731
    pic = lambda k, alt: ch.picture(charts[k], rel, alt) if charts.get(k) else None  # noqa: E731
    lim = enforced_limit(unl)
    by_id = {r["test_id"]: r for r in rows}
    L = []
    w = L.append
    w(f"# {g.get('name', 'GPU')}{' (' + system + ')' if system else ''} – gpu-unlock benchmark: stock vs. unlocked")
    w("")
    w(f"*Generated {now_iso()} by the benchmark suite v{SUITE_VERSION}. "
      f"Baseline run `{base['name']}`, unlocked run `{unl['name']}`.*")
    w("")
    n_err = sum(1 for c in checks if not c["ok"] and c["severity"] == "error")
    if n_err:
        w(f"> **Warning:** {n_err} validity check(s) failed. The numbers below are reported "
          "as measured, but the comparison may be affected – see *Validity checks*.")
        w("")

    # --- headline
    w("## Summary")
    w("")
    w("| Workload | Baseline | Unlocked | Change |")
    w("|---|---:|---:|---:|")
    for tid, label in HEADLINE:
        r = by_id.get(tid)
        if r and "change_pct" in r:
            w(f"| {label} | {fmt(r['median_b'])} {r['unit']} | {fmt(r['median_u'])} {r['unit']} | "
              f"**{pct(r['change_pct'])}** |")
    w("")
    w(_conclusions(rows))
    w("")
    if pic("headline", "Throughput gain per workload"):
        w(pic("headline", "Throughput gain per workload"))
        w("")
    eff = _efficiency_rows(by_id)
    if eff:
        sus = by_id.get("sustained_gemm_bf16_8192") or eff[0]
        w(f"**Efficiency:** the GPU draws the same power in both phases (≈ {fmt(sus.get('power_b'), 0)} W vs. "
          f"{fmt(sus.get('power_u'), 0)} W under load, enforced limit {fmt(lim, 0)} W) but finishes "
          f"{sus['change_pct']:.0f} % more work, so energy per unit of work drops by "
          f"{abs(sus['energy_change_pct']):.0f} %. See [Efficiency](#efficiency).")
        w("")

    # --- efficiency
    if eff:
        w("## Efficiency")
        w("")
        w("A laptop GPU runs at its power cap under sustained tensor load. gpu-unlock does not raise that cap: "
          "the card draws the same watts, runs at a *lower* SM clock, and still finishes more work, because each "
          "tensor-core instruction now completes at the full rate. More work for the same energy means higher "
          "efficiency: less energy per token, image or matrix, less heat for the same job, and on battery more "
          "work per charge.")
        w("")
        w("| Workload | Power stock → unlocked (W) | SM clock stock → unlocked (MHz) | Work per watt | Energy per unit of work |")
        w("|---|---|---|---:|---:|")
        for r in eff:
            w(f"| {ch.label(r['test_id'])} | {fmt(r['power_b'], 0)} → {fmt(r['power_u'], 0)} | "
              f"{fmt(r['clock_b'], 0)} → {fmt(r['clock_u'], 0)} | **{pct(r['ppw_change_pct'], 0)}** | "
              f"**{pct(r['energy_change_pct'], 0)}** |")
        w("")
        p = by_id.get("llm_prefill_bf16_p2048")
        if p and p.get("power_b") and p.get("power_u"):
            wh_b = p["power_b"] / p["median_b"] * 1e6 / 3600
            wh_u = p["power_u"] / p["median_u"] * 1e6 / 3600
            w(f"Example: processing one million prompt tokens (2048-token prompts, BF16) takes "
              f"{fmt(wh_b, 2)} Wh of GPU energy on the stock card and {fmt(wh_u, 2)} Wh unlocked.")
            w("")
        if pic("efficiency", "Throughput, power and clock change"):
            w(pic("efficiency", "Throughput, power and clock change"))
            w("")

    # --- system
    w("## Test system")
    w("")
    w("| | |")
    w("|---|---|")
    rows_sys = [
        ("GPU", f"{g.get('name')} ({g.get('pci.device_id')}, subsystem {g.get('pci.sub_device_id')})"),
        ("SMs / VRAM", f"{eu['torch']['sm_count']} SMs, {eu['torch']['total_mem_mib']} MiB"),
        ("VBIOS", g.get("vbios_version")),
        ("Power limit", f"{fmt(lim, 0)} W enforced under load (NVML; nvidia-smi current limit {g.get('power.limit')}, "
                        f"default {g.get('power.default_limit')}, "
                        f"max {g.get('power.max_limit')}), Dynamic Boost {eu.get('notebook_dynamic_boost')}, "
                        f"nvidia-powerd {eu.get('nvidia_powerd')}"),
        ("Host", f"{eu.get('cpu')}, {round((eu.get('ram_bytes') or 0) / 2**30)} GiB RAM"),
        ("OS / kernel", f"{eu.get('os')}, {eu.get('kernel')}"),
        ("NVIDIA driver", f"{g.get('driver_version')} ({eu.get('driver_flavour')} kernel modules)"),
        ("Software", f"PyTorch {eu['torch']['torch']}, CUDA {eu['torch']['cuda']}, cuDNN {eu['torch']['cudnn']}, "
                     f"Python {eu['torch']['python']}" + (f", torchvision {eu['torch']['torchvision']}"
                                                          if eu["torch"].get("torchvision") else "")),
        ("gpu-unlock", f"{eu['gpu_unlock_checkout'].get('describe') or 'unknown'} "
                       f"(version {eu['gpu_unlock_checkout'].get('version', '?')})"),
        ("Conditions", f"platform profile `{eu.get('platform_profile')}`, AC power {eu.get('ac_online')}, "
                       f"NVIDIA Runtime D3 `{eu.get('nvidia_runtime_d3')}`"),
        ("Idle reference", f"baseline {fmt(base['idle'].get('power_avg_w'), 1)} W / "
                           f"{fmt(base['idle'].get('temp_avg_c'), 0)} °C, unlocked "
                           f"{fmt(unl['idle'].get('power_avg_w'), 1)} W / {fmt(unl['idle'].get('temp_avg_c'), 0)} °C"),
    ]
    for k, v in rows_sys:
        w(f"| {k} | {v} |")
    w("")

    # --- validity
    w("## Validity checks")
    w("")
    w("| Check | Result | Detail |")
    w("|---|---|---|")
    for c in checks:
        mark = "pass" if c["ok"] else ("**FAIL**" if c["severity"] == "error" else "warn")
        w(f"| {c['check']} | {mark} | {c['detail']} |")
    w("")
    w("### Numerical correctness")
    w("")
    w("A fixed-seed 2048³ GEMM is run twice in each phase. Its output is hashed (SHA-256) and compared "
      "with an FP64 reference computed on the GPU.")
    w("")
    w("| Test | Bit-identical across phases | Deterministic within run (B / U) | Max rel. error vs FP64 (B / U) |")
    w("|---|---|---|---|")
    for c in corr:
        w(f"| {c['test_id']} | {'yes' if c['identical'] else 'no'} (`{c['sha_b']}` / `{c['sha_u']}`) | "
          f"{c['det_b']} / {c['det_u']} | {sci(c['err_b'])} / {sci(c['err_u'])} |")
    w("")

    # --- methodology
    a = eu["run"]["args"]
    w("## Method")
    w("")
    w(f"- Both phases run the identical test list on the same machine and software stack. The only "
      "difference is `gpu_unlock.targets=` on the kernel command line (the module then unlocks the card "
      "before the unmodified NVIDIA driver binds it).")
    w(f"- Every throughput test: {a['warmup_sec']:g} s warm-up, then {a['samples']} samples of "
      f"~{a['target_sec']:g} s each, timed with CUDA events. Reported value = median of the samples; "
      "the min/max range of all samples is given as well.")
    w(f"- Before each test the GPU idles until it is at or below {a['cooldown_temp']:g} °C "
      f"(at least {a['cooldown_min']:g} s, at most {a['cooldown_max']:g} s), so both phases start each test "
      "from a comparable thermal state.")
    w("- GPU power, clocks, temperature and NVML throttle reasons are sampled at 10 Hz during every "
      "measurement window (`telemetry.csv`). Implausible power readings (above 1.5× the maximum power limit) "
      "are discarded and counted.")
    w("- *Sep.* (separated) means the sample sets of both phases do not overlap at all (for 5 vs 5 samples an "
      "exact two-sided Mann–Whitney p of 0.008). The samples of one phase come from a single boot and are not "
      "fully independent, so this captures run-to-run noise, not boot-to-boot variation; repeating each phase "
      "on a second boot is the stronger test.")
    w("- FP16/BF16 GEMMs run with reduced-precision split-K reductions disabled, so accumulation is FP32 "
      "throughout. cuDNN autotuning (`cudnn.benchmark`) is on in both phases.")
    w("- Expectation labels: **gain** = path the unlock targets (FP16/BF16/TF32 tensor math with FP32 "
      "accumulation); **partial** = workloads that only partly use that path; **control** = must not change "
      "(FP32 on CUDA cores, INT8 tensor math, memory bandwidth); **info** = expected roughly unchanged but "
      "sensitive to CPU/launch overhead (batch-1 token generation). Control tests check that nothing else "
      "changed between the two boots.")
    w("- LLM tests use a synthetic Llama-style ~0.95 B parameter model with random weights, so no model "
      "download is needed; throughput does not depend on weight values.")
    w("")

    # --- results
    w("## Results")
    if pic("overview", "Change per test"):
        w("")
        w(pic("overview", "Change per test"))
    for cat in CAT_ORDER:
        rr = [r for r in rows if r["category"] == cat]
        if not rr:
            continue
        w("")
        w(f"### {CAT_TITLE[cat]}")
        w("")
        w("| Test | Expect | Baseline | Unlocked | Change | Range | Sep. | Power B→U (W) | Perf/W | "
          "SM clock B→U (MHz) | Max °C B→U |")
        w("|---|---|---:|---:|---:|---|:-:|---|---:|---|---|")
        for r in rr:
            if "change_pct" not in r:
                w(f"| {r['test_id']} | {r['expectation']} | {r['status_b']} | {r['status_u']} | – | – | – | – | – | – | – |")
                continue
            w(f"| {r['test_id']} | {r['expectation']} | {fmt(r['median_b'])} | {fmt(r['median_u'])} {r['unit']} | "
              f"**{pct(r['change_pct'])}** | {pct(r['range_lo_pct'])} … {pct(r['range_hi_pct'])} | "
              f"{'yes' if r['separated'] else 'no'} | {fmt(r['power_b'], 0)} → {fmt(r['power_u'], 0)} | "
              f"{pct(r.get('ppw_change_pct'))} | {fmt(r['clock_b'], 0)} → {fmt(r['clock_u'], 0)} | "
              f"{fmt(r['temp_b'], 0)} → {fmt(r['temp_u'], 0)} |")
        if cat == "gemm" and pic("gemm", "GEMM throughput"):
            w("")
            w(pic("gemm", "GEMM throughput"))
    w("")
    w("Test descriptions:")
    w("")
    for r in rows:
        w(f"- `{r['test_id']}` – {r['desc']}")
    w("")

    # --- sustained & power
    s = by_id.get("sustained_gemm_bf16_8192")
    tb, tu = base["tests"].get("sustained_gemm_bf16_8192"), unl["tests"].get("sustained_gemm_bf16_8192")
    if s and tb and tu and tb.get("extra") and tu.get("extra"):
        w("## Sustained load, power and thermals")
        w("")
        xb, xu = tb["extra"], tu["extra"]
        w("| | Baseline | Unlocked |")
        w("|---|---:|---:|")
        w(f"| First 5 s window (TFLOPS) | {fmt(xb.get('first_window'))} | {fmt(xu.get('first_window'))} |")
        w(f"| Median after 30 s (TFLOPS) | {fmt(xb.get('steady_median_after_30s'))} | {fmt(xu.get('steady_median_after_30s'))} |")
        w(f"| Median of last 60 s (TFLOPS) | {fmt(xb.get('last_60s_median'))} | {fmt(xu.get('last_60s_median'))} |")
        w(f"| Average GPU power (W) | {fmt(s.get('power_b'), 1)} | {fmt(s.get('power_u'), 1)} |")
        w(f"| Average SM clock (MHz) | {fmt(s.get('clock_b'), 0)} | {fmt(s.get('clock_u'), 0)} |")
        w(f"| Max GPU temperature (°C) | {fmt(s.get('temp_b'), 0)} | {fmt(s.get('temp_u'), 0)} |")
        w(f"| TFLOPS per watt | {fmt(s.get('ppw_b'), 3)} | {fmt(s.get('ppw_u'), 3)} |")
        thr = lambda d: ", ".join(f"{k} {v}%" for k, v in d.items()) or "none"  # noqa: E731
        w(f"| Throttle reasons (share of samples) | {thr(s.get('throttle_b', {}))} | {thr(s.get('throttle_u', {}))} |")
        w("")
        if pic("sustained", "Sustained run"):
            w(pic("sustained", "Sustained run"))
            w("")

    # --- reproducibility
    if repeats:
        w("## Reproducibility")
        w("")
        w("Additional complete runs of the same phase, compared with the runs used above (median per run):")
        w("")
        names = [base["name"], unl["name"]] + [r["name"] for r in repeats]
        w("| Test | " + " | ".join(f"`{n}`" for n in names) + " | Spread of unlocked runs |")
        w("|---|" + "---:|" * len(names) + "---:|")
        for tid, lab in HEADLINE:
            vals = []
            for run in [base, unl] + list(repeats):
                t = run["tests"].get(tid)
                vals.append(t["stats"].get("median") if t and t.get("status") == "ok" else None)
            if vals[0] is None:
                continue
            ul = [v for v, run in zip(vals, [base, unl] + list(repeats)) if v is not None and run["env"]["run"]["phase"] == "unlocked"]
            spread = (max(ul) / min(ul) - 1) * 100 if len(ul) > 1 else None
            w(f"| {lab} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(spread, 1, ' %')} |")
        w("")

    # --- caveats + raw data
    w("## Limitations")
    w("")
    w("- One laptop, one sample of silicon. Laptop results depend strongly on the power limit and cooling; "
      "desktop cards with higher power limits reach larger gains.")
    w("- gpu-unlock is experimental third-party software. It changes undocumented hardware registers; "
      "use it at your own risk.")
    w("- Gaming performance was not measured; the unlock does not touch the FP32 CUDA-core path games mainly use.")
    w("")
    w("## Raw data")
    w("")
    for run in (base, unl):
        rel = rel_results(run["path"])
        w(f"- `{run['name']}`: [run.log]({rel}/run.log), [results.jsonl]({rel}/results.jsonl), "
          f"[results.csv]({rel}/results.csv), [telemetry.csv]({rel}/telemetry.csv), [env.json]({rel}/env.json), "
          f"[raw environment]({rel}/env/), [manifest.json]({rel}/manifest.json) (SHA-256 of every file)")
    w("- This comparison: [comparison.csv](comparison.csv), [comparison.json](comparison.json), "
      "[compare.log](compare.log)")
    w("")
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


EFF_IDS = ["gemm_bf16_8192", "gemm_fp16_8192", "gemm_tf32_8192", "sdpa_fwdbwd_bf16", "llm_prefill_bf16_p2048",
           "train_gpt124m_bf16", "resnet50_train_bf16", "sustained_gemm_bf16_8192"]


def _efficiency_rows(by_id):
    out = []
    for i in EFF_IDS:
        r = by_id.get(i)
        if r and r.get("ppw_change_pct") is not None and r.get("clock_b"):
            r["energy_change_pct"] = (1 / (1 + r["ppw_change_pct"] / 100) - 1) * 100
            out.append(r)
    return out


def enforced_limit(run):
    vals = [t["telemetry"].get("power_limit_w") for t in run["tests"].values()
            if t.get("status") == "ok" and (t.get("telemetry") or {}).get("power_limit_w")]
    return describe(vals).get("median") if vals else None


def _conclusions(rows) -> str:
    by = {r["test_id"]: r for r in rows if "change_pct" in r}
    parts = []
    gain = [r for r in by.values() if r["expectation"] == "gain" and r["category"] == "gemm"]
    if gain:
        best = max(gain, key=lambda r: r["change_pct"])
        parts.append(f"The largest GEMM gain is **{best['change_pct']:+.1f} %** (`{best['test_id']}`).")
    ctrl = [r for r in by.values() if r["expectation"] == "control"]
    if ctrl:
        worst = max(ctrl, key=lambda r: abs(r["change_pct"]))
        parts.append(f"Control tests moved by at most {abs(worst['change_pct']):.1f} % (`{worst['test_id']}`), "
                     "so the two boots are otherwise comparable.")
    wl = [r for r in by.values() if r["category"] == "workload"]
    if wl:
        txt = ", ".join(f"`{r['test_id']}` {r['change_pct']:+.1f} %" for r in sorted(wl, key=lambda r: -r["change_pct"]))
        parts.append(f"End-to-end workloads: {txt}.")
    return " ".join(parts)


def write_compat(out, base, unl, rows, system=None):
    eu = unl["env"]
    g = eu["gpu"]
    by = {r["test_id"]: r for r in rows}
    gain = by.get("gemm_bf16_8192", {}).get("change_pct")
    result = ("Unlocked, and FP16/BF16 got faster" if eu["gpu_unlock"]["state"] == "unlocked" and (gain or 0) > 5
              else "Unlock completed, but no speedup" if eu["gpu_unlock"]["state"] == "unlocked"
              else "Unlock failed (error in the log)" if eu["gpu_unlock"]["state"] == "failed"
              else "gpu_unlock never ran (no gpu_unlock lines in the log)")
    klog = open(os.path.join(unl["path"], "env", "gpu_unlock-kernel-log.log"), encoding="utf-8").read().split("\n", 2)[-1].strip()
    lspci = eu.get("lspci", "")
    pw = by.get("gemm_bf16_8192", {}).get("power_u")
    L = [
        "# Compatibility report (paste into the gpu-unlock issue form)", "",
        f"**Title:** Compatibility report: {g.get('name')} ({_pci_id(lspci)})", "",
        f"**Result:** {result}", "",
        f"**Card:** {g.get('name')} {round(unl['env']['torch']['total_mem_mib'] / 1024)} GB in {system or '<laptop model> (fill in)'}", "",
        "**PCI IDs:**", "```text", lspci, "```", "",
        f"**VBIOS version:** {g.get('pci.bus_id')}, {g.get('vbios_version')}", "",
        "**Is the card modified?** Stock", "",
        f"**Distribution:** {eu.get('os')}", "",
        f"**Kernel:** {eu.get('kernel')}", "",
        f"**NVIDIA driver:** {g.get('driver_version')}, {eu.get('driver_flavour')} kernel modules", "",
        "**Initramfs generator:** initramfs-tools", "",
        f"**gpu-unlock version and install method:** {eu['gpu_unlock_checkout'].get('commit') or '?'}, make install", "",
        f"**Kernel command line:** `{eu.get('cmdline')}`", "",
        "**Setup:**", "```text",
        f"Laptop (Optimus, iGPU drives the display), single GPU, enforced power limit {fmt(enforced_limit(unl), 0)} W "
        f"(max {g.get('power.max_limit')}), Dynamic Boost {eu.get('notebook_dynamic_boost')}, "
        f"nvidia-powerd {eu.get('nvidia_powerd')}, NVIDIA Runtime D3 {eu.get('nvidia_runtime_d3')}, "
        f"Secure Boot: {eu.get('secure_boot') or 'unknown'}", "```", "",
        "**Kernel log:**", "```text", klog, "```", "",
        "**Benchmark before and after:**", "```text",
        f"torch {eu['torch']['torch']} cuBLAS (CUDA {eu['torch']['cuda']}), 8192³ dense GEMM, FP32 accumulation, "
        f"median of {eu['run']['args']['samples']} samples, ~{fmt(pw, 0)} W measured",
        "```", "",
        "| Input | Before | After | Change |", "| ----- | ------ | ----- | ------ |",
    ]
    for k in ("fp16", "bf16", "tf32"):
        r = by.get(f"gemm_{k}_8192")
        if r and "change_pct" in r:
            L.append(f"| {k.upper()} | {r['median_b']:.2f} TFLOPS | {r['median_u']:.2f} TFLOPS | {pct(r['change_pct'])} |")
    for tid, label in (("llm_prefill_bf16_p2048", "LLM prefill 2048 tok"), ("llm_decode_bf16_b1", "LLM decode b1"),
                       ("train_gpt124m_bf16", "GPT-124M train BF16")):
        r = by.get(tid)
        if r and "change_pct" in r:
            L.append(f"| {label} | {r['median_b']:.1f} {r['unit']} | {r['median_u']:.1f} {r['unit']} | {pct(r['change_pct'])} |")
    L += ["", "Full report and raw logs: <link to your repository>"]
    with open(os.path.join(out, "compat_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def _pci_id(lspci: str) -> str:
    m = re.search(r"\[(10de:[0-9a-f]{4})\]", lspci)
    return m.group(1) if m else "10de:????"


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", help="baseline run directory (default: newest)")
    ap.add_argument("--unlocked", help="unlocked run directory (default: newest)")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "reports"))
    ap.add_argument("--control-tolerance", type=float, default=5.0, help="max |change| %% for control tests")
    ap.add_argument("--system", default=None, help='device name for titles, e.g. "HP Omen 16-k0620ng"')
    ap.add_argument("--no-repeats", action="store_true", help="do not list other runs of the same phase")
    args = ap.parse_args()

    bdir = args.baseline or (find_runs("baseline") or [None])[-1]
    udir = args.unlocked or (find_runs("unlocked") or [None])[-1]
    if not bdir or not udir:
        print("Need one baseline and one unlocked run in results/ (smoke runs are ignored).", file=sys.stderr)
        sys.exit(2)
    base, unl = load_run(os.path.abspath(bdir)), load_run(os.path.abspath(udir))
    out = os.path.join(args.out_root, f"{time.strftime('%Y%m%d-%H%M%S')}_{base['name']}_vs_{unl['name']}")
    os.makedirs(out, exist_ok=True)
    log = setup_logging(os.path.join(out, "compare.log"), "compare")
    log.info("Comparing baseline %s with unlocked %s", base["name"], unl["name"])

    ids = list(dict.fromkeys(list(unl["tests"]) + list(base["tests"])))
    rows = [compare_test(base["tests"].get(t), unl["tests"].get(t)) for t in ids]
    for r in rows:
        if "change_pct" in r:
            log.info("%-28s %10.2f -> %10.2f %-9s %+7.2f %%  range %+.1f..%+.1f %%  %s", r["test_id"],
                     r["median_b"], r["median_u"], r["unit"], r["change_pct"], r["range_lo_pct"],
                     r["range_hi_pct"], "significant" if r["separated"] else "overlap")
        else:
            log.warning("%-28s not comparable (baseline %s, unlocked %s)", r["test_id"], r["status_b"], r["status_u"])
    checks = validity(base, unl, rows, args.control_tolerance)
    for c in checks:
        (log.info if c["ok"] else log.warning)("CHECK %-45s %s  %s", c["check"], "pass" if c["ok"] else "FAIL", c["detail"])
    corr = correctness_rows(base, unl)
    for c in corr:
        log.info("CORRECTNESS %-18s identical=%s err_b=%s err_u=%s", c["test_id"], c["identical"], c["err_b"], c["err_u"])

    cdir = os.path.join(out, "charts")
    os.makedirs(cdir, exist_ok=True)
    charts = {}
    try:
        sub = (f"{args.system + ', ' if args.system else ''}{unl['env']['gpu'].get('name')}, both phases at the same "
               f"power cap (≈ {fmt(enforced_limit(unl), 0)} W)")
        charts["headline"] = ch.headline(rows, cdir, [t for t, _ in HEADLINE if t != "llm_decode_bf16_b1"], sub)
        charts["overview"] = ch.overview(rows, cdir)
        charts["gemm"] = ch.gemm(rows, cdir)
        charts["efficiency"] = ch.efficiency(rows, cdir, EFF_IDS)
        charts["sustained"] = ch.sustained(base["path"], unl["path"], cdir)
    except ImportError:
        log.warning("matplotlib not installed – charts skipped")
    repeats = []
    if not args.no_repeats:
        for d in find_runs("unlocked") + find_runs("baseline"):
            if os.path.abspath(d) not in (base["path"], unl["path"]):
                repeats.append(load_run(os.path.abspath(d)))
    rel_results = lambda p: os.path.relpath(p, out)  # noqa: E731
    write_report(out, base, unl, rows, checks, corr, charts, rel_results, args.system, repeats)
    write_compat(out, base, unl, rows, args.system)

    flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows]
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with open(os.path.join(out, "comparison.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(flat)
    with open(os.path.join(out, "comparison.json"), "w", encoding="utf-8") as fh:
        json.dump({"generated": now_iso(), "suite_version": SUITE_VERSION, "baseline": base["name"],
                   "unlocked": unl["name"], "checks": checks, "correctness": corr, "tests": rows}, fh, indent=2)
    log.info("Report written to %s", os.path.join(out, "report.md"))


if __name__ == "__main__":
    main()
