"""Shared infrastructure for the RTX 3070 Ti gpu-unlock benchmark suite.

Logging, GPU telemetry sampling (NVML), robust statistics and small helpers.
Nothing in here imports torch, so compare.py can reuse it on any machine.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

SUITE_VERSION = "1.0.0"

# NVML clock-event ("throttle") reason bits, see nvml.h
THROTTLE_BITS = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000002: "app_clocks_setting",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "sw_thermal",
    0x0000000000000040: "hw_thermal",
    0x0000000000000080: "hw_power_brake",
    0x0000000000000100: "display_clock_setting",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def decode_throttle(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


# --------------------------------------------------------------------------- logging

def setup_logging(log_path: str, name: str = "bench") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    def _excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook
    return logger


class JsonlWriter:
    """Append-only JSON-lines file, flushed after every record (crash-safe)."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        record = {"ts": now_iso(), **record}
        with self._lock:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- shell

def sh(cmd: str, timeout: float = 30.0) -> tuple[int, str]:
    """Run a shell command, never raise. Returns (returncode, combined output)."""
    try:
        p = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "LC_ALL": "C"},
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return 127, f"<error: {e}>"


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- statistics

def describe(values: list[float]) -> dict:
    v = sorted(float(x) for x in values if x is not None and not math.isnan(x))
    n = len(v)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(v)
    sd = statistics.stdev(v) if n > 1 else 0.0
    if n >= 4:
        q1, _, q3 = statistics.quantiles(v, n=4, method="inclusive")
    else:
        q1, q3 = v[0], v[-1]
    return {
        "n": n,
        "median": statistics.median(v),
        "mean": mean,
        "stdev": sd,
        "cv_pct": (sd / mean * 100.0) if mean else 0.0,
        "min": v[0],
        "max": v[-1],
        "q1": q1,
        "q3": q3,
    }


# --------------------------------------------------------------------------- telemetry

class Telemetry:
    """Background sampler for power, clocks, temperature and throttle reasons.

    Uses NVML (nvidia-ml-py) at `interval` seconds; falls back to nvidia-smi
    polling at 1 s, or disables itself (CPU dry runs). Every sample is written
    to telemetry.csv and kept in memory so a test can aggregate exactly the
    samples taken during its measurement window.
    """

    FIELDS = [
        "ts", "t_rel_s", "test_id", "power_w", "power_limit_w", "sm_clock_mhz",
        "mem_clock_mhz", "temp_c", "gpu_util_pct", "mem_used_mib", "throttle_mask",
        "throttle_reasons", "power_valid",
    ]

    def __init__(self, csv_path: str | None, gpu_index: int = 0, interval: float = 0.1,
                 power_sanity_max_w: float = 400.0, log: logging.Logger | None = None):
        self.log = log or logging.getLogger("bench")
        self.interval = interval
        self.power_sanity_max_w = power_sanity_max_w
        self.tag = "idle"
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t0 = time.time()
        self.backend = None
        self._nv = None
        self._h = None
        try:
            import pynvml  # provided by nvidia-ml-py
            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.backend = "nvml"
        except Exception as e:  # noqa: BLE001
            if have("nvidia-smi"):
                self.backend = "nvidia-smi"
                self.interval = max(interval, 1.0)
                self.log.warning("NVML unavailable (%s); falling back to nvidia-smi polling at 1 s", e)
            else:
                self.log.warning("No NVML and no nvidia-smi: telemetry disabled (dry run?)")
        self._csv_fh = None
        self._csv = None
        if csv_path and self.backend:
            self._csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
            self._csv = csv.DictWriter(self._csv_fh, fieldnames=self.FIELDS)
            self._csv.writeheader()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)

    # -- sampling
    def _sample_nvml(self) -> dict:
        nv, h = self._nv, self._h
        s: dict = {}

        def safe(fn, *a):
            try:
                return fn(h, *a)
            except Exception:  # noqa: BLE001
                return None

        p = safe(nv.nvmlDeviceGetPowerUsage)
        s["power_w"] = p / 1000.0 if p is not None else None
        pl = safe(nv.nvmlDeviceGetEnforcedPowerLimit)
        s["power_limit_w"] = pl / 1000.0 if pl is not None else None
        s["sm_clock_mhz"] = safe(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_SM)
        s["mem_clock_mhz"] = safe(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_MEM)
        s["temp_c"] = safe(nv.nvmlDeviceGetTemperature, nv.NVML_TEMPERATURE_GPU)
        util = safe(nv.nvmlDeviceGetUtilizationRates)
        s["gpu_util_pct"] = util.gpu if util is not None else None
        mem = safe(nv.nvmlDeviceGetMemoryInfo)
        s["mem_used_mib"] = round(mem.used / 2**20) if mem is not None else None
        fn = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or getattr(
            nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        mask = safe(fn) if fn else None
        s["throttle_mask"] = mask
        return s

    def _sample_smi(self) -> dict:
        q = ("power.draw,enforced.power.limit,clocks.sm,clocks.mem,temperature.gpu,"
             "utilization.gpu,memory.used,clocks_event_reasons.active")
        rc, out = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits -i 0", timeout=5)
        if rc != 0:
            rc, out = sh(f"nvidia-smi --query-gpu={q.replace('clocks_event_reasons', 'clocks_throttle_reasons')}"
                         " --format=csv,noheader,nounits -i 0", timeout=5)
        vals = [x.strip() for x in out.splitlines()[0].split(",")] if rc == 0 and out else []

        def num(i, cast=float):
            try:
                return cast(vals[i])
            except Exception:  # noqa: BLE001
                return None

        mask = None
        if len(vals) > 7:
            try:
                mask = int(vals[7], 16)
            except ValueError:
                mask = None
        return {
            "power_w": num(0), "power_limit_w": num(1), "sm_clock_mhz": num(2, int),
            "mem_clock_mhz": num(3, int), "temp_c": num(4, int), "gpu_util_pct": num(5, int),
            "mem_used_mib": num(6, int), "throttle_mask": mask,
        }

    def sample_now(self) -> dict:
        if self.backend == "nvml":
            return self._sample_nvml()
        if self.backend == "nvidia-smi":
            return self._sample_smi()
        return {}

    def _run(self) -> None:
        last_flush = time.time()
        while not self._stop.is_set():
            t = time.time()
            try:
                s = self.sample_now()
                p = s.get("power_w")
                s["power_valid"] = p is not None and 0.0 <= p <= self.power_sanity_max_w
                mask = s.get("throttle_mask")
                s["throttle_reasons"] = "|".join(decode_throttle(mask)) if mask is not None else ""
                s["ts"] = now_iso()
                s["t_rel_s"] = round(t - self._t0, 3)
                s["test_id"] = self.tag
                with self._lock:
                    self.samples.append(s)
                    if self._csv:
                        self._csv.writerow({k: s.get(k) for k in self.FIELDS})
                        if t - last_flush > 1.0:
                            self._csv_fh.flush()
                            last_flush = t
            except Exception as e:  # noqa: BLE001  (never let the sampler die silently)
                self.log.debug("telemetry sample failed: %s", e)
            self._stop.wait(max(0.0, self.interval - (time.time() - t)))

    # -- control
    def start(self) -> "Telemetry":
        if self.backend:
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        if self._thread.is_alive():
            self.log.warning("telemetry thread did not stop; leaving files open")
            return
        if self._csv_fh:
            with self._lock:
                self._csv_fh.flush()
                self._csv_fh.close()
        if self.backend == "nvml":
            try:
                self._nv.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    def set_tag(self, tag: str) -> None:
        self.tag = tag

    def mark(self) -> int:
        with self._lock:
            return len(self.samples)

    def window(self, start_idx: int, end_idx: int | None = None) -> dict:
        """Aggregate samples between two marks (measurement window only)."""
        with self._lock:
            win = self.samples[start_idx:end_idx]
        if not win:
            return {"samples": 0}
        pw = [s["power_w"] for s in win if s.get("power_valid")]
        invalid = sum(1 for s in win if s.get("power_w") is not None and not s.get("power_valid"))
        clk = [s["sm_clock_mhz"] for s in win if s.get("sm_clock_mhz")]
        mclk = [s["mem_clock_mhz"] for s in win if s.get("mem_clock_mhz")]
        tmp = [s["temp_c"] for s in win if s.get("temp_c") is not None]
        util = [s["gpu_util_pct"] for s in win if s.get("gpu_util_pct") is not None]
        mem = [s["mem_used_mib"] for s in win if s.get("mem_used_mib") is not None]
        lim = [s["power_limit_w"] for s in win if s.get("power_limit_w")]
        reasons: dict[str, int] = {}
        for s in win:
            for r in filter(None, (s.get("throttle_reasons") or "").split("|")):
                reasons[r] = reasons.get(r, 0) + 1
        n = len(win)
        return {
            "samples": n,
            "power_avg_w": statistics.fmean(pw) if pw else None,
            "power_median_w": statistics.median(pw) if pw else None,
            "power_max_w": max(pw) if pw else None,
            "power_invalid_samples": invalid,
            "power_limit_w": statistics.median(lim) if lim else None,
            "sm_clock_avg_mhz": statistics.fmean(clk) if clk else None,
            "sm_clock_min_mhz": min(clk) if clk else None,
            "mem_clock_avg_mhz": statistics.fmean(mclk) if mclk else None,
            "temp_avg_c": statistics.fmean(tmp) if tmp else None,
            "temp_max_c": max(tmp) if tmp else None,
            "gpu_util_avg_pct": statistics.fmean(util) if util else None,
            "mem_used_max_mib": max(mem) if mem else None,
            "throttle_share_pct": {k: round(v / n * 100, 1) for k, v in sorted(reasons.items())},
        }

    def current_temp(self) -> float | None:
        try:
            return self.sample_now().get("temp_c")
        except Exception:  # noqa: BLE001
            return None
