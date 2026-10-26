"""Report charts for the gpu-unlock benchmark, rendered in a light and a dark variant.

Every chart is written as <name>_light.png and <name>_dark.png. report.md / README.md embed them with
<picture>, so GitHub shows the variant that matches the reader's theme.

Colours: validated categorical slots (blue, orange, aqua) from the dataviz reference palette, light and
dark steps; text and grid use neutral text tokens, never series colours.
"""
from __future__ import annotations

import csv
import os

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3de",
              "axis": "#a3a29c", "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "grid": "#33332f",
             "axis": "#6b6a64", "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70"},
}
# Semantic roles: unlocked = slot 1 (blue), baseline = slot 2 (orange)
UNL, BASE = "blue", "orange"

LABELS = {
    "sdpa_fwd_fp16": "Attention forward (FP16)",
    "sdpa_fwd_bf16": "Attention forward (BF16)",
    "sdpa_fwdbwd_bf16": "Attention fwd+bwd (BF16)",
    "conv3x3_fp16": "3×3 convolution (FP16)",
    "conv3x3_bf16": "3×3 convolution (BF16)",
    "llm_prefill_bf16_p512": "LLM prompt processing, 512 tok",
    "llm_prefill_bf16_p2048": "LLM prompt processing, 2048 tok",
    "llm_decode_bf16_b1": "LLM generation, batch 1",
    "llm_decode_bf16_b16": "LLM generation, batch 16",
    "train_gpt124m_bf16": "GPT-124M training (BF16)",
    "train_gpt124m_fp16": "GPT-124M training (FP16)",
    "resnet50_train_bf16": "ResNet-50 training (BF16)",
    "resnet50_infer_fp16": "ResNet-50 inference (FP16)",
    "sd_pipeline_bf16_512": "Stable Diffusion 512² (BF16)",
    "sustained_gemm_bf16_8192": "Sustained BF16 load, 3 min",
    "membw_copy_1gib": "Memory bandwidth (1 GiB copy)",
}
GROUPS = [
    ("Matrix math (FP32 accumulate)", ["gemm_bf16_", "gemm_fp16_", "gemm_tf32_"]),
    ("Kernels", ["sdpa_", "conv3x3_"]),
    ("Workloads", ["llm_", "train_", "resnet50_", "sd_pipeline_"]),
    ("Sustained", ["sustained_"]),
    ("Controls – must stay at 0 %", ["gemm_fp32_", "gemm_int8_", "membw_"]),
]


def label(tid: str) -> str:
    if tid in LABELS:
        return LABELS[tid]
    parts = tid.split("_")
    if parts[0] == "gemm" and len(parts) == 3:
        return f"{parts[1].upper()} matmul {parts[2]}³"
    return tid


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(plt, t):
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["text"], "axes.labelcolor": t["text2"], "xtick.color": t["text2"], "ytick.color": t["text"],
        "axes.edgecolor": t["axis"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "figure.dpi": 140, "legend.frameon": False, "legend.labelcolor": t["text"],
    })


def _save(fig, out_dir, name, theme):
    p = os.path.join(out_dir, f"{name}_{theme}.png")
    fig.savefig(p, bbox_inches="tight", pad_inches=0.25)
    return p


def _pct(v):
    return f"{v:+.0f} %" if abs(v) >= 9.95 else f"{v:+.1f} %"


# ---------------------------------------------------------------------------------------------- charts

def headline(rows, out_dir, ids, subtitle):
    """Hero chart: % gain of the key workloads (single series, direct labels, no legend)."""
    data = [(label(r["test_id"]), r["change_pct"]) for i in ids for r in rows
            if r["test_id"] == i and "change_pct" in r]
    if not data:
        return None
    plt = _plt()
    out = {}
    for th, t in THEMES.items():
        _style(plt, t)
        fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(data) + 1.3))
        y = list(range(len(data)))
        vals = [v for _, v in data]
        ax.barh(y, vals, height=0.62, color=t[UNL], edgecolor=t["surface"], linewidth=2)
        for i, v in enumerate(vals):
            ax.text(max(v, 0) + 1.0, i, _pct(v), va="center", ha="left", color=t["text"], fontsize=10,
                    fontweight="bold")
        ax.set_yticks(y, [n for n, _ in data])
        ax.invert_yaxis()
        ax.axvline(0, color=t["axis"], lw=1)
        ax.set_xlim(min(0, min(vals)) - 2, max(vals) * 1.18 + 2)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Speed-up with gpu-unlock vs. stock (median, %)")
        ax.set_title("Throughput gain with gpu-unlock", pad=22)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=t["text2"], fontsize=9, va="bottom")
        out[th] = _save(fig, out_dir, "headline", th)
        plt.close(fig)
    return out


def overview(rows, out_dir):
    """All tests, grouped with headers; line = min/max range of all samples."""
    ok = {r["test_id"]: r for r in rows if "change_pct" in r}
    order, headers = [], []
    used = set()
    for title, prefixes in GROUPS:
        ids = [i for i in ok if any(i.startswith(p) for p in prefixes) and i not in used]
        if title.startswith("Controls"):
            ids = [i for i in ok if ok[i]["expectation"] == "control" and i not in used]
        if not ids:
            continue
        headers.append((len(order), title))
        dorder = ["bf16", "fp16", "tf32", "fp32", "int8"]

        def key(x):
            p = x.split("_")
            if p[0] == "gemm" and len(p) == 3:
                return (0, dorder.index(p[1]) if p[1] in dorder else 9, int(p[2]))
            return (1, list(ok).index(x), 0)   # suite order for everything else
        for i in sorted(ids, key=key):
            order.append(i)
            used.add(i)
    if not order:
        return None
    plt = _plt()
    out = {}
    for th, t in THEMES.items():
        _style(plt, t)
        n = len(order) + len(headers)
        fig, ax = plt.subplots(figsize=(9, 0.3 * n + 1.4))
        ypos, y, ticks, tlabels = {}, 0, [], []
        for idx, tid in enumerate(order):
            for hpos, title in headers:
                if hpos == idx:
                    ax.text(-0.01, y, title, transform=ax.get_yaxis_transform(), ha="right", va="center",
                            fontweight="bold", color=t["text"], fontsize=9.5)
                    y += 1
            ypos[tid] = y
            ticks.append(y)
            tlabels.append(label(tid))
            y += 1
        for tid in order:
            r, yy = ok[tid], ypos[tid]
            ax.barh(yy, r["change_pct"], height=0.66, color=t[UNL], edgecolor=t["surface"], linewidth=2)
            ax.plot([r["range_lo_pct"], r["range_hi_pct"]], [yy, yy], color=t["text2"], lw=1.2,
                    solid_capstyle="butt")
            ax.text(max(r["change_pct"], r["range_hi_pct"], 0) + 1.2, yy, _pct(r["change_pct"]), va="center",
                    fontsize=8.5, color=t["text"])
        ax.set_yticks(ticks, tlabels, fontsize=8.5)
        ax.set_ylim(y - 0.4, -0.8)
        ax.axvline(0, color=t["axis"], lw=1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Change unlocked vs. stock (median, %); thin line = min…max range of all samples")
        ax.set_title("Every test at a glance")
        out[th] = _save(fig, out_dir, "overview_change", th)
        plt.close(fig)
    return out


def gemm(rows, out_dir):
    ok = [r for r in rows if r["category"] == "gemm" and "change_pct" in r]
    if not ok:
        return None
    kinds = [k for k in ("bf16", "fp16", "tf32", "fp32", "int8") if any(r["test_id"].startswith(f"gemm_{k}_") for r in ok)]
    plt = _plt()
    out = {}
    for th, t in THEMES.items():
        _style(plt, t)
        fig, axes = plt.subplots(1, len(kinds), figsize=(2.75 * len(kinds), 3.6))
        for ax, k in zip(axes, kinds):
            rr = sorted((r for r in ok if r["test_id"].startswith(f"gemm_{k}_")), key=lambda r: int(r["test_id"].rsplit("_", 1)[1]))
            xi = range(len(rr))
            ax.bar([i - 0.2 for i in xi], [r["median_b"] for r in rr], 0.4, color=t[BASE], edgecolor=t["surface"], linewidth=2, label="stock")
            ax.bar([i + 0.2 for i in xi], [r["median_u"] for r in rr], 0.4, color=t[UNL], edgecolor=t["surface"], linewidth=2, label="unlocked")
            top = max(max(r["median_b"], r["median_u"]) for r in rr)
            for i, r in enumerate(rr):
                ax.text(i, max(r["median_b"], r["median_u"]) + top * 0.03, _pct(r["change_pct"]), ha="center",
                        fontsize=8.5, color=t["text"], fontweight="bold" if r["expectation"] == "gain" else "normal")
            ax.set_ylim(0, top * 1.18)
            ax.set_xticks(list(xi), [r["test_id"].rsplit("_", 1)[1] for r in rr])
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{k.upper()}" + (" – control" if rr[0]["expectation"] == "control" else ""), fontsize=10.5)
            ax.set_ylabel(rr[0]["unit"])
            ax.set_xlabel("matrix size N")
        h_, l_ = axes[0].get_legend_handles_labels()
        fig.suptitle("Dense matrix multiply throughput (median)", x=0.01, ha="left", fontweight="bold", color=t["text"])
        fig.legend(h_, l_, loc="upper right", ncols=2, fontsize=9, bbox_to_anchor=(0.995, 1.0))
        fig.tight_layout()
        out[th] = _save(fig, out_dir, "gemm_throughput", th)
        plt.close(fig)
    return out


def efficiency(rows, out_dir, ids):
    """Indexed to stock = 0 %: throughput vs. average power vs. SM clock (one shared % axis)."""
    data = [r for i in ids for r in rows if r["test_id"] == i and "change_pct" in r and r.get("power_b") and r.get("clock_b")]
    if not data:
        return None
    series = [("Throughput", UNL, lambda r: r["change_pct"]),
              ("Average GPU power", BASE, lambda r: (r["power_u"] / r["power_b"] - 1) * 100),
              ("Average SM clock", "aqua", lambda r: (r["clock_u"] / r["clock_b"] - 1) * 100)]
    plt = _plt()
    out = {}
    for th, t in THEMES.items():
        _style(plt, t)
        fig, ax = plt.subplots(figsize=(9, 0.95 * len(data) + 1.6))
        h = 0.26
        allv = []
        for j, (name, col, fn) in enumerate(series):
            ys = [i + (j - 1) * h for i in range(len(data))]
            vs = [fn(r) for r in data]
            allv += vs
            ax.barh(ys, vs, height=h, color=t[col], edgecolor=t["surface"], linewidth=1.5, label=name)
            for yy, v in zip(ys, vs):  # direct labels on every bar (relief rule for the aqua slot)
                ax.text(v + (1.0 if v >= 0 else -1.0), yy, _pct(v), va="center", ha="left" if v >= 0 else "right",
                        fontsize=8, color=t["text"])
        ax.set_yticks(range(len(data)), [label(r["test_id"]) for r in data])
        ax.invert_yaxis()
        ax.axvline(0, color=t["axis"], lw=1)
        ax.set_xlim(min(allv) - 12, max(allv) + 12)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Change unlocked vs. stock (%)")
        ax.set_title("Same power, lower clock, more work", pad=22)
        ax.text(0, 1.02, "gpu-unlock raises work done per clock cycle; the GPU stays at its power cap",
                transform=ax.transAxes, color=t["text2"], fontsize=9, va="bottom")
        ax.legend(loc="lower right", fontsize=8.5)
        out[th] = _save(fig, out_dir, "efficiency", th)
        plt.close(fig)
    return out


def sustained(base_path, unl_path, out_dir, test_id="sustained_gemm_bf16_8192"):
    def load(p):
        f = os.path.join(p, f"sustained_{test_id}.csv")
        return list(csv.DictReader(open(f, encoding="utf-8"))) if os.path.exists(f) else []
    tb, tu = load(base_path), load(unl_path)
    if not (tb and tu):
        return None

    def col(tl, k):
        out = []
        for r in tl:
            try:
                out.append(float(r[k]))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out
    plt = _plt()
    out = {}
    panels = [("value", "TFLOPS"), ("power_avg_w", "GPU power (W)"), ("sm_clock_avg_mhz", "SM clock (MHz)")]
    for th, t in THEMES.items():
        _style(plt, t)
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for ax, (k, yl) in zip(axes, panels):
            for tl, c, name in ((tb, BASE, "stock"), (tu, UNL, "unlocked")):
                xs, ys = col(tl, "t_s"), col(tl, k)
                ax.plot(xs, ys, color=t[c], lw=2, label=name)
                if k != "power_avg_w":   # power lines overlap by design – annotated once instead
                    ax.text(xs[-1] + 3, ys[-1], name, color=t["text"], fontsize=8.5, va="center")
            if k == "power_avg_w":
                pv = [v for v in col(tb, k) + col(tu, k) if v == v]
                ax.text(0.99, 0.2, f"both ≈ {sorted(pv)[len(pv) // 2]:.0f} W – lines overlap", transform=ax.transAxes,
                        ha="right", color=t["text2"], fontsize=9)
            ax.set_ylabel(yl)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylim(bottom=0)
        axes[1].set_ylim(bottom=0)
        axes[2].set_ylim(bottom=0)
        axes[-1].set_xlabel("time (s), 5 s windows")
        axes[0].set_title("Sustained BF16 load: more TFLOPS at the same power", pad=10)
        axes[0].legend(loc="lower right", fontsize=8.5, ncols=2)
        fig.tight_layout()
        out[th] = _save(fig, out_dir, "sustained_bf16", th)
        plt.close(fig)
    return out


def picture(paths: dict, rel, alt: str) -> str:
    """GitHub-compatible <picture> that follows the reader's light/dark theme."""
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{rel(paths["dark"])}">\n'
            f'  <img alt="{alt}" src="{rel(paths["light"])}">\n</picture>')


# ---------------------------------------------------------------------------------------------- llama.cpp series

def llama_change(rows, out_dir):
    """Series 2: % change per weight format; panels = flash attention off/on, bars = pp512 / pp2048 / tg."""
    ok = [r for r in rows if "change_pct" in r]
    if not ok:
        return None
    quants = list(dict.fromkeys(r["extra_quant"] for r in ok))
    tests = list(dict.fromkeys(r["test_label"] for r in ok))
    names = {t: (f"prompt {t[2:]} tok" if t.startswith("pp") else f"generation {t[2:]} tok") for t in tests}
    cols = [UNL, BASE, "aqua"]
    plt = _plt()
    out = {}
    for th, t in THEMES.items():
        _style(plt, t)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
        allv = [r["change_pct"] for r in ok]
        for ax, fa in zip(axes, ("off", "on")):
            w = 0.8 / max(1, len(tests))
            for j, tl in enumerate(tests):
                xs, vs = [], []
                for i, q in enumerate(quants):
                    r = next((r for r in ok if r["extra_quant"] == q and r["test_label"] == tl and r["extra_fa"] == fa), None)
                    if r:
                        xs.append(i + (j - (len(tests) - 1) / 2) * w)
                        vs.append(r["change_pct"])
                ax.bar(xs, vs, w, color=t[cols[j % 3]], edgecolor=t["surface"], linewidth=1.5, label=names[tl])
                for x, v in zip(xs, vs):  # compact labels: bars are narrow
                    ax.text(x, v + (0.8 if v >= 0 else -0.8), f"{v:+.0f}" if abs(v) >= 1 else f"{v:+.1f}",
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=t["text"])
            ax.axhline(0, color=t["axis"], lw=1)
            ax.set_xticks(range(len(quants)), quants)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"flash attention {fa}", fontsize=10.5)
        axes[0].set_ylabel("change unlocked vs. stock (%), labels in %")
        axes[0].set_ylim(min(0, min(allv)) - 5, max(allv) * 1.15 + 5)
        h_, l_ = axes[0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="upper right", ncols=len(tests), fontsize=9, bbox_to_anchor=(0.995, 1.0))
        fig.suptitle("llama.cpp: effect of gpu-unlock per weight format", x=0.01, ha="left", fontweight="bold",
                     color=t["text"])
        fig.tight_layout()
        out[th] = _save(fig, out_dir, "llamacpp_change", th)
        plt.close(fig)
    return out


def llama_abs(rows, out_dir, test_label="pp2048", fa="on"):
    """Absolute tokens/s, stock vs. unlocked, for one test across weight formats."""
    rr = [r for r in rows if "change_pct" in r and r["test_label"] == test_label and r["extra_fa"] == fa]
    if not rr:
        return None
    plt = _plt()
    out = {}
    for th, t in THEMES.items():
        _style(plt, t)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        xi = range(len(rr))
        ax.bar([i - 0.2 for i in xi], [r["median_b"] for r in rr], 0.4, color=t[BASE], edgecolor=t["surface"], linewidth=2, label="stock")
        ax.bar([i + 0.2 for i in xi], [r["median_u"] for r in rr], 0.4, color=t[UNL], edgecolor=t["surface"], linewidth=2, label="unlocked")
        top = max(max(r["median_b"], r["median_u"]) for r in rr)
        for i, r in enumerate(rr):
            ax.text(i + 0.2, r["median_u"] + top * 0.02, _pct(r["change_pct"]), ha="center", fontsize=9,
                    color=t["text"], fontweight="bold")
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks(list(xi), [r["extra_quant"] for r in rr])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("tokens/s")
        kind = f"prompt processing, {test_label[2:]} tokens" if test_label.startswith("pp") else f"generation, {test_label[2:]} tokens"
        ax.set_title(f"llama.cpp {kind} (flash attention {fa})")
        ax.legend(loc="upper left", fontsize=9, ncols=2)
        out[th] = _save(fig, out_dir, f"llamacpp_{test_label}_fa-{fa}", th)
        plt.close(fig)
    return out
