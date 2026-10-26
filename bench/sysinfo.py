"""Environment capture and unlock-state detection.

Every raw command output is stored verbatim under <run>/env/*.log so a reader
can verify the summary in env.json against the source.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import re

from common import sh, have

# Commands captured verbatim. None of them needs root.
RAW_COMMANDS = {
    "uname": "uname -a",
    "os-release": "cat /etc/os-release",
    "cmdline": "cat /proc/cmdline",
    "lscpu": "lscpu",
    "meminfo": "free -b",
    "lspci-nvidia": "lspci -nn -D -d 10de:",
    "lspci-nvidia-verbose": "lspci -vv -nn -D -d 10de: 2>/dev/null | head -120",
    "nvidia-smi-q": "nvidia-smi -q",
    "nvidia-smi-power": "nvidia-smi -q -d POWER,CLOCK,PERFORMANCE,TEMPERATURE",
    "nvidia-driver-version": "cat /proc/driver/nvidia/version",
    "nvidia-driver-params": "cat /proc/driver/nvidia/params",
    "nvidia-gpu-power": "cat /proc/driver/nvidia/gpus/*/power",
    "nvidia-modinfo": "modinfo nvidia 2>/dev/null | head -20",
    "lsmod-gpu": "lsmod | grep -E '^(gpu_unlock|nvidia|nouveau)'",
    "gpu_unlock-modinfo": "modinfo gpu_unlock 2>/dev/null",
    "gpu_unlock-kernel-log": "journalctl -k -b --no-pager -o short-iso 2>/dev/null | grep -i gpu_unlock"
                             " || dmesg 2>/dev/null | grep -i gpu_unlock",
    "dkms-status": "dkms status 2>/dev/null",
    "platform-profile": "cat /sys/firmware/acpi/platform_profile /sys/firmware/acpi/platform_profile_choices",
    "power-profiles-daemon": "powerprofilesctl get 2>/dev/null",
    "nvidia-powerd": "systemctl is-active nvidia-powerd; journalctl -u nvidia-powerd -b --no-pager 2>/dev/null | tail -20",
    "ac-adapter": "for d in /sys/class/power_supply/*; do echo \"$d type=$(cat $d/type) online=$(cat $d/online 2>/dev/null)\"; done",
    "battery": "for d in /sys/class/power_supply/BAT*; do echo \"$d status=$(cat $d/status) capacity=$(cat $d/capacity)\"; done",
    "gpu-compute-apps": "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv",
    "secure-boot": "mokutil --sb-state 2>/dev/null",
}


def _first(pattern: str, text: str, flags: int = re.M) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def capture(env_dir: str, gpu_unlock_dir: str | None = None) -> dict:
    os.makedirs(env_dir, exist_ok=True)
    raw: dict[str, str] = {}
    for name, cmd in RAW_COMMANDS.items():
        rc, out = sh(cmd, timeout=30)
        raw[name] = out
        with open(os.path.join(env_dir, f"{name}.log"), "w", encoding="utf-8") as fh:
            fh.write(f"$ {cmd}\n# exit={rc}\n{out}\n")

    q = ("name,pci.bus_id,pci.device_id,pci.sub_device_id,vbios_version,driver_version,"
         "power.limit,power.default_limit,power.min_limit,power.max_limit,"
         "clocks.max.sm,clocks.max.mem,memory.total,pcie.link.gen.max,pcie.link.width.max")
    rc, out = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader -i 0")
    gpu = {}
    if rc == 0 and out:
        vals = [v.strip() for v in out.splitlines()[0].split(",")]
        gpu = dict(zip(q.split(","), vals))
    with open(os.path.join(env_dir, "nvidia-smi-query.log"), "w", encoding="utf-8") as fh:
        fh.write(f"# {q}\n{out}\n")

    bus = (gpu.get("pci.bus_id") or "").lower()
    # nvidia-smi prints 8-digit domains (00000000:01:00.0); the kernel uses 4 (0000:01:00.0)
    bus_kernel = re.sub(r"^0000(?=[0-9a-f]{4}:)", "", bus)

    # gpu-unlock checkout
    gu = {"dir": gpu_unlock_dir, "commit": None, "describe": None}
    if gpu_unlock_dir and os.path.isdir(os.path.join(gpu_unlock_dir, ".git")):
        gu["commit"] = sh(f"git -C '{gpu_unlock_dir}' rev-parse --short HEAD")[1] or None
        gu["describe"] = sh(f"git -C '{gpu_unlock_dir}' log -1 --format='%h %cs %s'")[1] or None
        vf = os.path.join(gpu_unlock_dir, "VERSION")
        if os.path.exists(vf):
            gu["version"] = open(vf, encoding="utf-8").read().strip()

    ac_online = ac_state()

    rtd3 = _first(r"Runtime D3 status:\s*(.+)", raw.get("nvidia-gpu-power", ""))
    dyn_boost = _first(r"Notebook Dynamic Boost:\s*(.+)", raw.get("nvidia-gpu-power", ""))

    env = {
        "host": platform.node(),
        "kernel": platform.release(),
        "os": _first(r'^PRETTY_NAME="?([^"\n]+)', raw["os-release"]),
        "cpu": _first(r"Model name:\s*(.+)", raw["lscpu"]),
        "ram_bytes": _int(_first(r"^Mem:\s+(\d+)", raw["meminfo"])),
        "cmdline": raw["cmdline"],
        "gpu": gpu,
        "gpu_bus_kernel": bus_kernel,
        "lspci": raw["lspci-nvidia"],
        "driver_flavour": "open" if "Open Kernel Module" in raw.get("nvidia-driver-version", "")
                          or "Dual MIT/GPL" in raw.get("nvidia-modinfo", "") else "proprietary/unknown",
        "nvidia_runtime_d3": rtd3,
        "notebook_dynamic_boost": dyn_boost,
        "nvidia_powerd": raw["nvidia-powerd"].splitlines()[0] if raw["nvidia-powerd"] else None,
        "platform_profile": _read("/sys/firmware/acpi/platform_profile"),
        "ac_online": ac_online,
        "secure_boot": raw.get("secure-boot") or None,
        "gpu_unlock_checkout": gu,
        "gpu_unlock": detect_unlock(raw, bus_kernel),
    }
    return env


def ac_state() -> bool | None:
    """True if a mains/USB-C power source is online. Ignores batteries of mice, keyboards etc."""
    state = None
    for d in glob.glob("/sys/class/power_supply/*"):
        if _read(os.path.join(d, "type")) not in ("Mains", "USB"):
            continue
        on = _read(os.path.join(d, "online"))
        if on is not None:
            state = bool(state) or on == "1"
    return state


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def detect_unlock(raw: dict, bus_kernel: str) -> dict:
    """Decide from cmdline + kernel log whether this boot has an unlocked card."""
    cmdline = raw.get("cmdline", "")
    log = raw.get("gpu_unlock-kernel-log", "")
    lsmod = raw.get("lsmod-gpu", "")
    opt_in = _first(r"gpu_unlock\.targets=(\S+)", cmdline)
    lines = [ln for ln in log.splitlines() if "gpu_unlock" in ln.lower()]
    mine = [ln for ln in lines if not bus_kernel or bus_kernel in ln.lower()] or lines
    unlocked = any(re.search(r"gpu_unlock: (unlocked|still unlocked|unlocked again)", ln) for ln in mine)
    # Only the module's own error messages count. Kernel notices such as
    # "gpu_unlock: module verification failed: signature ... - tainting kernel" are not unlock failures.
    benign = ("module verification failed", "taints kernel", "tainting kernel")
    failed = [ln for ln in mine if re.search(r"gpu_unlock: (unlock|unlock after resume|firmware load) failed", ln)
              and not any(b in ln for b in benign)]
    if unlocked and not failed:
        state = "unlocked"
    elif failed:
        state = "failed"
    elif opt_in and not lines:
        state = "unknown"  # opted in but log unreadable/empty
    else:
        state = "stock"
    return {
        "state": state,
        "cmdline_targets": opt_in,
        "module_loaded": bool(re.search(r"^gpu_unlock\b", lsmod, re.M)),
        "log_lines": len(lines),
        "log_unlocked": unlocked,
        "log_failures": failed[:5],
    }


def quick_state() -> dict:
    """Unlock state of the current boot without writing anything (used by --phase auto / status)."""
    raw = {k: sh(RAW_COMMANDS[k])[1] for k in ("cmdline", "gpu_unlock-kernel-log", "lsmod-gpu")}
    bus = sh("nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i 0")[1].strip().lower()
    return detect_unlock(raw, re.sub(r"^0000(?=[0-9a-f]{4}:)", "", bus))


def status() -> int:
    """Human-readable preflight status, used by `./run.sh status`."""
    st = quick_state()
    rtd3 = _first(r"Runtime D3 status:\s*(.+)", sh(RAW_COMMANDS["nvidia-gpu-power"])[1])
    q = sh("nvidia-smi --query-gpu=name,pci.bus_id,power.limit,power.max_limit,temperature.gpu "
           "--format=csv,noheader")[1]
    items = [
        ("GPU", q),
        ("gpu_unlock state", f"{st['state']} (cmdline targets={st['cmdline_targets']}, module loaded="
                             f"{st['module_loaded']}, log lines={st['log_lines']})"),
        ("platform_profile", _read("/sys/firmware/acpi/platform_profile")),
        ("AC adapter online", ac_state()),
        ("NVIDIA Runtime D3", rtd3),
        ("nvidia-powerd", sh("systemctl is-active nvidia-powerd")[1]),
        ("GPU compute processes", sh("nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader")[1] or "none"),
    ]
    for k, v in items:
        print(f"{k:24s} {v}")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        sys.exit(status())
    if len(sys.argv) > 1 and sys.argv[1] == "state":
        print(quick_state()["state"])
        sys.exit(0)
    print(json.dumps(capture(sys.argv[1] if len(sys.argv) > 1 else "/tmp/env-test"), indent=2))
