#!/usr/bin/env python3
"""Compare a baseline and an unlocked llama.cpp run (test series 2) and write a publishable report.

  python3 bench/llamacpp_compare.py                  # newest non-smoke runs in results/llamacpp/
  python3 bench/llamacpp_compare.py --baseline DIR --unlocked DIR

Output: reports/llamacpp/<timestamp>_<baseline>_vs_<unlocked>/
  report.md, comparison.csv, comparison.json, compare.log, charts/*.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SUITE_VERSION, describe, now_iso, read_jsonl, setup_logging  # noqa: E402
from compare import compare_test, fmt, pct  # noqa: E402
import charts as chm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "llamacpp")
QUANT_NOTE = {
    "BF16": "cuBLAS, BF16 inputs with **FP32 accumulation** – the path gpu-unlock targets",
    "F16": "cuBLAS, FP16 inputs with **FP16 accumulation** – not rate-limited on GeForce",
    "Q8_0": "MMQ kernels, INT8 tensor-core math",
    "Q4_K_M": "MMQ kernels, INT8 tensor-core math (4-bit weights unpacked to INT8)",
}


def newest(phase):
    runs = sorted(d for d in glob.glob(os.path.join(RES, f"*_{phase}*"))
                  if "_smoke" not in os.path.basename(d) and os.path.exists(os.path.join(d, "results.jsonl")))
    return runs[-1] if runs else None


def load(path):
    recs = read_jsonl(os.path.join(path, "results.jsonl"))
    return {"path": path, "name": os.path.basename(path),
            "env": json.load(open(os.path.join(path, "env.json"), encoding="utf-8")),
            "tests": {r["test_id"]: r for r in recs if r.get("type") == "test_result"},
            "idle": next((r.get("telemetry", {}) for r in recs if r.get("type") == "idle_reference"), {})}


def checks(base, unl, rows, tol):
    eb, eu = base["env"], unl["env"]
    out = []

    def add(name, ok, detail, sev="error"):
        out.append({"check": name, "ok": bool(ok), "detail": detail, "severity": sev})

    add("Baseline boot is stock", eb["gpu_unlock"]["state"] == "stock", eb["gpu_unlock"]["state"])
    add("Unlocked boot reports 'gpu_unlock: unlocked'", eu["gpu_unlock"]["state"] == "unlocked", eu["gpu_unlock"]["state"])
    add("Same llama.cpp build", eb["llamacpp"]["commit"] == eu["llamacpp"]["commit"],
        f"{eb['llamacpp']['commit']} / {eu['llamacpp']['commit']}")
    mb, mu = eb["llamacpp"]["models"], eu["llamacpp"]["models"]
    same_models = all(mb.get(k, {}).get("sha256") == v.get("sha256") for k, v in mu.items())
    add("Identical model files (SHA-256)", same_models, ", ".join(sorted(mu)))
    for key, label in [("driver_version", "NVIDIA driver"), ("vbios_version", "VBIOS"), ("power.max_limit", "max power limit")]:
        add(f"Same {label}", eb["gpu"].get(key) == eu["gpu"].get(key), f"{eb['gpu'].get(key)} / {eu['gpu'].get(key)}")
    for key in ("kernel", "platform_profile", "ac_online", "nvidia_runtime_d3"):
        add(f"Same {key}", eb.get(key) == eu.get(key), f"{eb.get(key)} / {eu.get(key)}")
    add("Same GGML environment overrides", eb["llamacpp"]["env_overrides"] == eu["llamacpp"]["env_overrides"],
        json.dumps(eu["llamacpp"]["env_overrides"]))
    add("No run forced past preflight", not (eb["run"].get("forced") or eu["run"].get("forced")),
        f"{eb.get('preflight_problems') or '-'} / {eu.get('preflight_problems') or '-'}", "warning")
    ngl = {(r["extra"].get("n_gpu_layers")) for run in (base, unl) for r in run["tests"].values() if r.get("extra")}
    add("All layers offloaded to GPU in every test", ngl <= {99, None}, f"n_gpu_layers seen: {sorted(map(str, ngl))}")
    ctrl = [r for r in rows if r["expectation"] == "control" and "change_pct" in r]
    bad = [r for r in ctrl if abs(r["change_pct"]) > tol]
    add(f"Token-generation controls unchanged (±{tol:g} %)", ctrl and not bad,
        ", ".join(f"{r['test_id']} {r['change_pct']:+.1f}%" for r in ctrl) or "no control results")
    cvs = [max(r["cv_b"], r["cv_u"]) for r in rows if "cv_b" in r]
    if cvs:
        add("Measurement noise acceptable (max CV < 5 %)", max(cvs) < 5,
            f"median CV {describe(cvs)['median']:.2f} %, worst {max(cvs):.2f} %", "warning")
    return out


def charts(rows, out_dir, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib missing – no charts")
        return {}
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
    ok = [r for r in rows if "change_pct" in r]
    quants = list(dict.fromkeys(r["extra_quant"] for r in ok))
    kinds = list(dict.fromkeys(r["test_label"] for r in ok))
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    colors = ["#2f6fdb", "#7aa5ea", "#b8bcc4", "#8a8f98", "#1b1f24"]
    for ax, fa in zip(axes, ("off", "on")):
        width = 0.8 / max(1, len(kinds))
        for j, k in enumerate(kinds):
            vals = []
            for q in quants:
                r = next((r for r in ok if r["extra_quant"] == q and r["test_label"] == k and r["extra_fa"] == fa), None)
                vals.append(r["change_pct"] if r else 0.0)
            xs = [i + (j - (len(kinds) - 1) / 2) * width for i in range(len(quants))]
            ax.bar(xs, vals, width, color=colors[j % len(colors)], label=k)
        ax.axhline(0, color="#1b1f24", lw=0.8)
        ax.set_xticks(range(len(quants)), quants)
        ax.set_title(f"flash attention {fa}")
    axes[0].set_ylabel("change unlocked vs. baseline (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("llama.cpp (llama-bench): effect of gpu-unlock per weight format")
    fig.tight_layout()
    p = os.path.join(out_dir, "llamacpp_change.png")
    fig.savefig(p)
    plt.close(fig)
    res["change"] = p
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline")
    ap.add_argument("--unlocked")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "reports", "llamacpp"))
    ap.add_argument("--control-tolerance", type=float, default=5.0)
    ap.add_argument("--system", default=None, help='device name for the title, e.g. "HP Omen 16-k0620ng"')
    ap.add_argument("--name", default=None, help="fixed report folder name, e.g. series2-llamacpp")
    a = ap.parse_args()
    bdir, udir = a.baseline or newest("baseline"), a.unlocked or newest("unlocked")
    if not bdir or not udir:
        print("Need one baseline and one unlocked llama.cpp run in results/llamacpp/.", file=sys.stderr)
        sys.exit(2)
    base, unl = load(os.path.abspath(bdir)), load(os.path.abspath(udir))
    out = os.path.join(a.out_root, a.name or f"{time.strftime('%Y%m%d-%H%M%S')}_{base['name']}_vs_{unl['name']}")
    os.makedirs(out, exist_ok=True)
    log = setup_logging(os.path.join(out, "compare.log"), "llamacpp-compare")
    log.info("Comparing %s with %s", base["name"], unl["name"])

    ids = list(dict.fromkeys(list(unl["tests"]) + list(base["tests"])))
    rows = []
    for t in ids:
        b, u = base["tests"].get(t), unl["tests"].get(t)
        r = compare_test(b, u)
        ex = (u or b).get("extra", {})
        fa = "on" if t.endswith("_fa-on") else "off" if t.endswith("_fa-off") else str(ex.get("flash_attn"))
        r["extra_quant"], r["extra_fa"], r["extra_kind"] = ex.get("quant"), fa, ex.get("kind")
        r["test_label"] = f"pp{ex.get('n_prompt')}" if ex.get("kind") == "pp" else f"tg{ex.get('n_gen')}"
        rows.append(r)
        if "change_pct" in r:
            log.info("%-26s %9.1f -> %9.1f tok/s %+7.2f %%  range %+.1f..%+.1f %%", t, r["median_b"], r["median_u"],
                     r["change_pct"], r["range_lo_pct"], r["range_hi_pct"])
        else:
            log.warning("%-26s not comparable (%s / %s)", t, r["status_b"], r["status_u"])
    ck = checks(base, unl, rows, a.control_tolerance)
    for c in ck:
        (log.info if c["ok"] else log.warning)("CHECK %-45s %s  %s", c["check"], "pass" if c["ok"] else "FAIL", c["detail"])
    cdir = os.path.join(out, "charts")
    os.makedirs(cdir, exist_ok=True)
    ch = {}
    try:
        ch["change"] = chm.llama_change(rows, cdir)
        ch["abs"] = chm.llama_abs(rows, cdir, "pp2048", "on")
    except ImportError:
        log.warning("matplotlib missing – no charts")
    rel = lambda p: os.path.relpath(p, out)  # noqa: E731
    pic = lambda k, alt: chm.picture(ch[k], rel, alt) if ch.get(k) else None  # noqa: E731

    eu, g = unl["env"], unl["env"]["gpu"]
    L = []
    w = L.append
    w(f"# {g.get('name')}{' (' + a.system + ')' if a.system else ''} – gpu-unlock and llama.cpp (GGUF models)")
    w("")
    w(f"*Generated {now_iso()}, suite v{SUITE_VERSION}. Baseline `{base['name']}`, unlocked `{unl['name']}`.*")
    w("")
    nfail = sum(1 for c in ck if not c["ok"] and c["severity"] == "error")
    if nfail:
        w(f"> **Warning:** {nfail} validity check(s) failed – see *Validity checks*.")
        w("")
    w("## Summary")
    w("")
    labels = list(dict.fromkeys(r["test_label"] for r in rows))
    quants = list(dict.fromkeys(r["extra_quant"] for r in rows))
    for fa in ("on", "off"):
        w(f"**Flash attention {fa}** – change in tokens/s (unlocked vs. baseline, median):")
        w("")
        w("| Weights | " + " | ".join(labels) + " |")
        w("|---|" + "---:|" * len(labels))
        for q in quants:
            cells = []
            for lab in labels:
                r = next((r for r in rows if r["extra_quant"] == q and r["test_label"] == lab and r["extra_fa"] == fa), None)
                cells.append(f"{pct(r['change_pct'])}" + ("" if r.get("separated") else " ¹") if r and "change_pct" in r else "–")
            w(f"| {q} | " + " | ".join(cells) + " |")
        w("")
    w("¹ sample ranges of both phases overlap (difference within run-to-run noise).")
    w("")
    if pic("change", "Change per weight format"):
        w(pic("change", "Change per weight format"))
        w("")
    if pic("abs", "Prompt processing tokens/s"):
        w(pic("abs", "Prompt processing tokens/s"))
        w("")
    w("How llama.cpp computes each weight format on this GPU (llama.cpp CUDA backend):")
    w("")
    for q in quants:
        w(f"- **{q}** – {QUANT_NOTE.get(q, 'see llama.cpp source')}")
    w("")
    w("## Test system")
    w("")
    w("| | |")
    w("|---|---|")
    for k, v in [
        ("GPU", f"{g.get('name')} ({g.get('pci.device_id')}), VBIOS {g.get('vbios_version')}, max power {g.get('power.max_limit')}"),
        ("Host", f"{eu.get('cpu')}, {eu.get('os')}, kernel {eu.get('kernel')}"),
        ("Driver", f"{g.get('driver_version')} ({eu.get('driver_flavour')})"),
        ("llama.cpp", f"{eu['llamacpp']['describe']}, built with {eu['llamacpp']['nvcc']} for sm_86"),
        ("Models", "; ".join(f"`{k}` {v['bytes'] / 1e9:.2f} GB" for k, v in eu["llamacpp"]["models"].items())),
        ("Conditions", f"profile `{eu.get('platform_profile')}`, AC {eu.get('ac_online')}, Runtime D3 `{eu.get('nvidia_runtime_d3')}`, "
                       f"nvidia-powerd {eu.get('nvidia_powerd')}"),
        ("gpu-unlock", eu["gpu_unlock_checkout"].get("describe") or "?"),
    ]:
        w(f"| {k} | {v} |")
    w("")
    w("## Validity checks")
    w("")
    w("| Check | Result | Detail |")
    w("|---|---|---|")
    for c in ck:
        w(f"| {c['check']} | {'pass' if c['ok'] else ('**FAIL**' if c['severity'] == 'error' else 'warn')} | {c['detail']} |")
    w("")
    args = eu["run"]["args"]
    w("## Method")
    w("")
    w(f"- One source model (`{eu['run']['args']['model_slug']}`) converted once to BF16 GGUF and quantised from it, so all "
      "variants are the same network. Both phases use the identical llama.cpp build and identical files (SHA-256 checked).")
    w(f"- Each test is one `llama-bench` call with `-ngl 99 -r {args['reps']}` (warm-up run included by llama-bench), "
      "flash attention off and on. The reported value is the median of the repetitions (`samples_ts`).")
    w(f"- Before each test the GPU cools to ≤ {args['cooldown_temp']:g} °C (min {args['cooldown_min']:g} s, max "
      f"{args['cooldown_max']:g} s). Power/clock/temperature are averaged over the whole llama-bench call, "
      "including model loading, so watts here are indicative only.")
    w("- *pp* = prompt processing (prefill), *tg* = token generation. Token generation is memory-bandwidth bound and "
      "serves as the control: it must not change.")
    w("")
    w("## All results")
    w("")
    w("| Test | Expect | Baseline tok/s | Unlocked tok/s | Change | Range | Sep. | Avg W B→U | Max °C B→U |")
    w("|---|---|---:|---:|---:|---|:-:|---|---|")
    for r in rows:
        if "change_pct" not in r:
            w(f"| {r['test_id']} | {r['expectation']} | {r['status_b']} | {r['status_u']} | – | – | – | – | – |")
            continue
        w(f"| {r['test_id']} | {r['expectation']} | {fmt(r['median_b'], 1)} | {fmt(r['median_u'], 1)} | "
          f"**{pct(r['change_pct'])}** | {pct(r['range_lo_pct'])} … {pct(r['range_hi_pct'])} | "
          f"{'yes' if r['separated'] else 'no'} | {fmt(r['power_b'], 0)} → {fmt(r['power_u'], 0)} | "
          f"{fmt(r['temp_b'], 0)} → {fmt(r['temp_u'], 0)} |")
    w("")
    w("## Raw data")
    w("")
    for run in (base, unl):
        rel = os.path.relpath(run["path"], out)
        w(f"- `{run['name']}`: [run.log]({rel}/run.log), [results.jsonl]({rel}/results.jsonl), [results.csv]({rel}/results.csv), "
          f"[telemetry.csv]({rel}/telemetry.csv), [llama-bench raw output]({rel}/raw/), [env.json]({rel}/env.json), "
          f"[manifest.json]({rel}/manifest.json)")
    w("")
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows]
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with open(os.path.join(out, "comparison.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(flat)
    with open(os.path.join(out, "comparison.json"), "w", encoding="utf-8") as fh:
        json.dump({"generated": now_iso(), "baseline": base["name"], "unlocked": unl["name"], "checks": ck, "tests": rows},
                  fh, indent=2)
    log.info("Report written to %s", os.path.join(out, "report.md"))


if __name__ == "__main__":
    main()
