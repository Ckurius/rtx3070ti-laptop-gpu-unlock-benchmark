#!/usr/bin/env python3
"""Test series 2 – llama.cpp (llama-bench) with real GGUF models, baseline vs. gpu_unlock.

The same model is tested as BF16, F16, Q8_0 and Q4_K_M. In current llama.cpp (CUDA backend, Ampere):
  * BF16 weights  -> cuBLAS with FP32 accumulation   -> the path gpu-unlock speeds up
  * F16 weights   -> cuBLAS with FP16 accumulation   -> not rate-limited on GeForce, little effect expected
  * Q8_0 / Q4_K_M -> MMQ kernels (INT8 tensor cores) -> little effect expected
  * token generation is memory-bandwidth bound      -> no effect expected (control)
Every test runs with flash attention off and on.

Output: results/llamacpp/<timestamp>_<phase>/ with run.log, env.json, env/, results.jsonl, results.csv,
telemetry.csv, raw/<test>.json (llama-bench JSON) + raw/<test>.stderr.log, summary.md, manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (SUITE_VERSION, JsonlWriter, Telemetry, describe, now_iso, setup_logging,  # noqa: E402
                    sh, sha256_file)
import sysinfo  # noqa: E402

SERIES = "llamacpp"
EXPECT = {  # (quant, test kind) -> expectation
    ("BF16", "pp"): "gain",
}


def expectation(quant: str, kind: str) -> str:
    if kind == "tg":
        return "control"
    return EXPECT.get((quant, kind), "info")


def build_tests(args) -> list[dict]:
    tests = []
    pps = [512] if args.smoke else [int(x) for x in args.prompts.split(",")]
    tg = 32 if args.smoke else args.gen
    for q in args.quants.split():
        model = os.path.join(args.models_dir, f"{args.model_slug}-{q}.gguf")
        for fa in ("off", "on"):
            for p in pps:
                tests.append({"id": f"{q}_pp{p}_fa-{fa}", "quant": q, "kind": "pp", "n_prompt": p, "n_gen": 0,
                              "fa": fa, "model": model})
            tests.append({"id": f"{q}_tg{tg}_fa-{fa}", "quant": q, "kind": "tg", "n_prompt": 0, "n_gen": tg,
                          "fa": fa, "model": model})
    return tests


def cooldown(tele: Telemetry, args, log, test_id: str):
    tele.set_tag(f"cooldown:{test_id}")
    t0 = time.time()
    time.sleep(args.cooldown_min)
    temp = tele.current_temp()
    while temp is not None and temp > args.cooldown_temp and time.time() - t0 < args.cooldown_max:
        time.sleep(1.0)
        temp = tele.current_temp()
    if temp is not None and temp > args.cooldown_temp:
        log.warning("%s: start temperature %s °C above target after %.0f s", test_id, temp, time.time() - t0)
    return temp


def run_test(t: dict, args, log, tele: Telemetry, raw_dir: str) -> dict:
    cmd = [args.llama_bench, "-m", t["model"], "-p", str(t["n_prompt"]), "-n", str(t["n_gen"]),
           "-r", str(args.reps), "-ngl", "99", "-fa", t["fa"], "-o", "json"]
    log.debug("%s: %s", t["id"], " ".join(cmd))
    start_temp = cooldown(tele, args, log, t["id"])
    tele.set_tag(t["id"])
    mark = tele.mark()
    t0 = time.time()
    env = {**os.environ}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=args.timeout)
    duration = time.time() - t0
    win = tele.window(mark)
    tele.set_tag("idle")
    with open(os.path.join(raw_dir, f"{t['id']}.json"), "w", encoding="utf-8") as fh:
        fh.write(p.stdout)
    with open(os.path.join(raw_dir, f"{t['id']}.stderr.log"), "w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(cmd)}\n# exit={p.returncode} duration={duration:.1f}s\n{p.stderr}")
    rec = {"type": "test_result", "suite_version": SUITE_VERSION, "series": SERIES, "test_id": t["id"],
           "category": f"llamacpp_{t['kind']}", "expectation": expectation(t["quant"], t["kind"]),
           "unit": "tokens/s", "status": "error", "batch": None, "start_temp_c": start_temp,
           "duration_s": duration, "telemetry": win, "samples": [], "stats": {"n": 0},
           "desc": (f"llama-bench {t['quant']} {'prompt processing' if t['kind'] == 'pp' else 'token generation'} "
                    f"{t['n_prompt'] or t['n_gen']} tokens, flash attention {t['fa']}, all layers on GPU"),
           "extra": {"quant": t["quant"], "kind": t["kind"], "n_prompt": t["n_prompt"], "n_gen": t["n_gen"],
                     "flash_attn": t["fa"], "model_file": os.path.basename(t["model"])},
           "perf_per_watt": None, "error": None}
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-6:])
        rec["error"] = f"exit {p.returncode}: {tail}"[:800]
        rec["status"] = "oom" if "out of memory" in p.stderr.lower() else "error"
        return rec
    try:
        data = json.loads(p.stdout)
        r = data[0] if isinstance(data, list) else data
    except (json.JSONDecodeError, IndexError) as e:
        rec["error"] = f"cannot parse llama-bench JSON: {e}"
        return rec
    samples = r.get("samples_ts") or ([r["avg_ts"]] if "avg_ts" in r else [])
    rec["samples"] = samples
    rec["stats"] = describe(samples)
    rec["status"] = "ok" if samples else "error"
    for k in ("build_commit", "build_number", "model_type", "model_size", "model_n_params", "n_batch", "n_ubatch",
              "type_k", "type_v", "flash_attn", "n_gpu_layers", "gpu_info", "backends", "avg_ts", "stddev_ts", "test_time"):
        if k in r:
            rec["extra"][k] = r[k]
    if rec["stats"].get("median") and win.get("power_avg_w"):
        rec["perf_per_watt"] = rec["stats"]["median"] / win["power_avg_w"]
    return rec


def phase_check(env: dict, args, log):
    state = env["gpu_unlock"]["state"]
    problems = []
    if state == "unknown" and not ({"adm", "systemd-journal"} & set(sh("id -nG")[1].split())):
        problems.append("kernel log not readable: sudo usermod -aG adm $USER, then log in again")
    if args.phase == "baseline" and state in ("unlocked", "unknown"):
        problems.append(f"phase 'baseline' but gpu_unlock state is '{state}'")
    if args.phase == "unlocked" and state != "unlocked":
        problems.append(f"phase 'unlocked' but gpu_unlock state is '{state}'")
    if env.get("ac_online") is False:
        problems.append("running on battery – connect the power adapter")
    if env.get("platform_profile") and env["platform_profile"] != "performance":
        problems.append(f"platform_profile is '{env['platform_profile']}', expected 'performance'")
    if (env.get("nvidia_runtime_d3") or "").lower().startswith("enabled"):
        problems.append("NVIDIA Runtime D3 is enabled (GPU power-off resets the unlock) – run ./run.sh prepare")
    for p in problems:
        (log.warning if args.smoke else log.error)("PREFLIGHT: %s", p)
    if problems and not args.smoke and not args.force:
        log.critical("Preflight failed. Fix the problems or re-run with --force.")
        sys.exit(3)
    env["preflight_problems"] = problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--phase", choices=["baseline", "unlocked", "auto"], required=True)
    ap.add_argument("--llama-bench", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--model-slug", required=True)
    ap.add_argument("--quants", default="BF16 F16 Q8_0 Q4_K_M")
    ap.add_argument("--prompts", default="512,2048", help="prompt lengths for pp tests")
    ap.add_argument("--gen", type=int, default=128, help="generated tokens for tg tests")
    ap.add_argument("--reps", type=int, default=7, help="llama-bench repetitions per test")
    ap.add_argument("--cooldown-temp", type=float, default=55.0)
    ap.add_argument("--cooldown-min", type=float, default=10.0)
    ap.add_argument("--cooldown-max", type=float, default=120.0)
    ap.add_argument("--idle-sec", type=float, default=15.0)
    ap.add_argument("--timeout", type=float, default=900.0, help="max seconds per llama-bench call")
    ap.add_argument("--only", default=None, help="regex on test ids")
    ap.add_argument("--out-root", default=os.path.join(root, "results", SERIES))
    ap.add_argument("--gpu-unlock-dir", default=os.path.expanduser("~/gpu-unlock"))
    ap.add_argument("--label", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.reps, args.cooldown_min, args.cooldown_max, args.idle_sec = 2, 0.0, 0.0, 2.0

    if args.phase == "auto":
        args.phase = "unlocked" if sysinfo.quick_state()["state"] == "unlocked" else "baseline"
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{args.phase}" + ("_smoke" if args.smoke else "") + (
        "_" + re.sub(r"[^A-Za-z0-9_-]+", "-", args.label) if args.label else "")
    run_dir = os.path.join(args.out_root, name)
    raw_dir = os.path.join(run_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    log = setup_logging(os.path.join(run_dir, "run.log"))
    log.info("=== llama.cpp series v%s | phase=%s%s ===", SUITE_VERSION, args.phase, " | SMOKE" if args.smoke else "")
    log.info("Run directory: %s", run_dir)
    log.info("Arguments: %s", json.dumps(vars(args)))

    if not os.path.exists(args.llama_bench):
        log.critical("llama-bench not found at %s – run ./llama.sh setup", args.llama_bench)
        sys.exit(2)
    tests = build_tests(args)
    if args.only:
        tests = [t for t in tests if re.search(args.only, t["id"])]
    missing = sorted({t["model"] for t in tests if not os.path.exists(t["model"])})
    if missing:
        log.critical("Model files missing: %s – run ./llama.sh setup", ", ".join(missing))
        sys.exit(2)

    env = sysinfo.capture(os.path.join(run_dir, "env"), args.gpu_unlock_dir)
    llama_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.llama_bench)))
    def ok_or_none(res):
        return res[1] if res[0] == 0 and res[1] else None
    env["llamacpp"] = {
        "commit": ok_or_none(sh(f"git -C '{llama_dir}' rev-parse --short HEAD")),
        "describe": ok_or_none(sh(f"git -C '{llama_dir}' log -1 --format='%h %cs %s'")),
        "nvcc": ok_or_none(sh("nvcc --version | tail -1")),
        "env_overrides": {k: os.environ.get(k) for k in ("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "GGML_CUDA_FORCE_MMQ",
                                                         "GGML_CUDA_FORCE_CUBLAS", "CUDA_VISIBLE_DEVICES")},
        "models": {},
    }
    sums_file = os.path.join(args.models_dir, "SHA256SUMS")
    sums = {}
    if os.path.exists(sums_file):
        for line in open(sums_file, encoding="utf-8"):
            parts = line.split()
            if len(parts) == 2:
                sums[os.path.basename(parts[1])] = parts[0]
    for m in sorted({t["model"] for t in tests}):
        b = os.path.basename(m)
        env["llamacpp"]["models"][b] = {"bytes": os.path.getsize(m), "sha256": sums.get(b) or sha256_file(m)}
    env["run"] = {"suite_version": SUITE_VERSION, "series": SERIES, "phase": args.phase, "smoke": args.smoke,
                  "label": args.label, "started": now_iso(), "args": vars(args), "run_dir": name}
    g = env["gpu"]
    log.info("GPU %s | VBIOS %s | driver %s | power max %s | unlock state %s", g.get("name"), g.get("vbios_version"),
             g.get("driver_version"), g.get("power.max_limit"), env["gpu_unlock"]["state"])
    log.info("llama.cpp %s | %s", env["llamacpp"]["describe"], env["llamacpp"]["nvcc"])
    for b, info in env["llamacpp"]["models"].items():
        log.info("Model %s: %.2f GB, sha256 %s…", b, info["bytes"] / 1e9, info["sha256"][:16])
    phase_check(env, args, log)
    env["run"]["forced"] = bool(env.get("preflight_problems")) and not args.smoke
    with open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, default=str)

    try:
        max_w = float((g.get("power.max_limit") or "").split()[0])
    except (ValueError, IndexError):
        max_w = 0.0
    tele = Telemetry(os.path.join(run_dir, "telemetry.csv"),
                     power_sanity_max_w=max(250.0, 1.5 * max_w) if max_w else 400.0, log=log).start()
    jsonl = JsonlWriter(os.path.join(run_dir, "results.jsonl"))
    jsonl.write({"type": "run_start", "env": env})
    tele.set_tag("idle_reference")
    mk = tele.mark()
    time.sleep(args.idle_sec)
    idle = tele.window(mk)
    tele.set_tag("idle")
    jsonl.write({"type": "idle_reference", "telemetry": idle})

    log.info("Running %d llama-bench tests (%d repetitions each)", len(tests), args.reps)
    rows = []
    t_start = time.time()
    for i, t in enumerate(tests, 1):
        log.info("[%d/%d] %s", i, len(tests), t["id"])
        rec = run_test(t, args, log, tele, raw_dir)
        jsonl.write(rec)
        st, w = rec["stats"], rec["telemetry"] or {}
        if rec["status"] == "ok":
            thr = w.get("throttle_share_pct") or {}
            log.info("%-26s %10.1f tokens/s  sd %5.1f  n=%d | %s W | %s MHz | %s °C max | %s", t["id"], st["median"],
                     st["stdev"], st["n"], _f(w.get("power_avg_w"), 1), _f(w.get("sm_clock_avg_mhz"), 0),
                     _f(w.get("temp_max_c"), 0), ", ".join(f"{k} {v}%" for k, v in thr.items()) or "no throttle")
        else:
            log.error("%-26s %s %s", t["id"], rec["status"].upper(), rec["error"])
        rows.append({"test_id": t["id"], "quant": t["quant"], "kind": t["kind"], "flash_attn": t["fa"],
                     "n_prompt": t["n_prompt"], "n_gen": t["n_gen"], "expectation": rec["expectation"],
                     "status": rec["status"], "n": st.get("n", 0), "median": st.get("median"),
                     "mean": st.get("mean"), "stdev": st.get("stdev"), "cv_pct": st.get("cv_pct"),
                     "min": st.get("min"), "max": st.get("max"), "power_avg_w": w.get("power_avg_w"),
                     "sm_clock_avg_mhz": w.get("sm_clock_avg_mhz"), "temp_max_c": w.get("temp_max_c"),
                     "start_temp_c": rec["start_temp_c"], "duration_s": rec["duration_s"], "error": rec["error"]})
    total = time.time() - t_start
    jsonl.write({"type": "run_end", "duration_s": total, "finished": now_iso()})
    jsonl.close()
    tele.stop()

    with open(os.path.join(run_dir, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["test_id"])
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# llama.cpp run {name}\n\n- Phase **{args.phase}**, gpu_unlock state **{env['gpu_unlock']['state']}**\n"
                 f"- llama.cpp {env['llamacpp']['describe']}\n- Duration {total / 60:.1f} min\n\n"
                 "| Test | Expect | Median tokens/s | CV % | n | Avg W | Max °C | Status |\n|---|---|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            fh.write(f"| {r['test_id']} | {r['expectation']} | {_f(r['median'], 1)} | {_f(r['cv_pct'], 1)} | {r['n']} | "
                     f"{_f(r['power_avg_w'], 1)} | {_f(r['temp_max_c'], 0)} | {r['status']} |\n")
    files = {}
    for rt, _, names in os.walk(run_dir):
        for n in sorted(names):
            pth = os.path.join(rt, n)
            files[os.path.relpath(pth, run_dir)] = {"bytes": os.path.getsize(pth), "sha256": sha256_file(pth)}
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"generated": now_iso(), "files": files}, fh, indent=2)
    bad = sum(1 for r in rows if r["status"] != "ok")
    log.info("=== Finished in %.1f min: %d ok, %d failed. Results: %s ===", total / 60, len(rows) - bad, bad, run_dir)


def _f(v, nd):
    return "n/a" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    main()
