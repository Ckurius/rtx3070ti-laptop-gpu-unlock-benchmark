#!/usr/bin/env bash
# Entry point for the RTX 3070 Ti gpu-unlock benchmark.
#
#   ./run.sh status            show unlock state, power profile, AC, Runtime D3
#   ./run.sh prepare           one-time: disable NVIDIA Runtime D3, performance profile (sudo, reboot after)
#   ./run.sh smoke             2-3 min functional check of every test (not comparable)
#   ./run.sh baseline          full run on the stock card              (~30-40 min)
#   ./run.sh install-unlock    clone + install gpu-unlock, add kernel parameter (sudo, reboot after)
#   ./run.sh unlocked          full run with gpu_unlock active          (~30-40 min)
#   ./run.sh compare           build the report from the newest baseline + unlocked runs
#   ./run.sh remove-unlock     remove the kernel parameter again (sudo, reboot after)
#
# Extra arguments are passed on to bench/suite.py or bench/compare.py.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
HERE="$PWD"
GPU_UNLOCK_DIR="${GPU_UNLOCK_DIR:-$HOME/gpu-unlock}"
mkdir -p logs results reports
cmd="${1:-help}"; shift || true

venv() {
  [ -x .venv/bin/python ] || ./setup_env.sh
  # shellcheck disable=SC1091
  . .venv/bin/activate
}

oplog() {  # log admin actions (sudo steps) to logs/
  OPLOG="logs/${1}_$(date +%Y%m%d-%H%M%S).log"
  exec > >(tee -a "$OPLOG") 2>&1
  echo "== $1 $(date -Iseconds), log: $OPLOG"
}

gpu_bus() {  # 00000000:01:00.0 -> 0000:01:00.0
  nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i 0 | tr 'A-F' 'a-f' | sed -E 's/^0000([0-9a-f]{4}:)/\1/'
}

prep_session() {
  powerprofilesctl set performance 2>/dev/null || echo "note: could not set power profile via powerprofilesctl"
  echo "platform_profile: $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo n/a)"
}

bench() {  # keep the machine awake: suspend would reset the unlock registers
  venv
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  prep_session
  if command -v systemd-inhibit >/dev/null; then
    systemd-inhibit --what=idle:sleep --who="3070ti-unlock" --why="GPU benchmark running" \
      python bench/suite.py --gpu-unlock-dir "$GPU_UNLOCK_DIR" "$@"
  else
    python bench/suite.py --gpu-unlock-dir "$GPU_UNLOCK_DIR" "$@"
  fi
}

case "$cmd" in
  status)
    python3 bench/sysinfo.py status ;;

  prepare)
    oplog prepare
    conf=/etc/modprobe.d/zz-nvidia-no-rtd3.conf  # "zz": loaded after Ubuntu's nvidia-runtimepm.conf
    if grep -qs "NVreg_DynamicPowerManagement=0x00" "$conf"; then
      echo "Runtime D3 is already disabled in $conf"
    else
      echo "options nvidia NVreg_DynamicPowerManagement=0x00" | sudo tee "$conf"
      sudo update-initramfs -u
      echo ">>> Reboot required for the Runtime D3 change."
    fi
    prep_session
    systemctl is-active nvidia-powerd || echo "note: nvidia-powerd is not active (Dynamic Boost)"
    ;;

  smoke)     bench --phase auto --smoke "$@" ;;
  baseline)  bench --phase baseline "$@" ;;
  unlocked)  bench --phase unlocked "$@" ;;

  compare)
    venv
    python bench/compare.py "$@" ;;

  install-unlock)
    oplog install-unlock
    if [ ! -d "$GPU_UNLOCK_DIR/.git" ]; then
      git clone https://github.com/Zagitta/gpu-unlock "$GPU_UNLOCK_DIR"
    fi
    git -C "$GPU_UNLOCK_DIR" log -1 --format='gpu-unlock commit %h (%cs) %s'
    (cd "$GPU_UNLOCK_DIR" && make install)
    bus="$(gpu_bus)"
    echo "GPU PCI address: $bus"
    if grep -q "gpu_unlock.targets=" /etc/default/grub; then
      echo "gpu_unlock.targets already present in /etc/default/grub"
    else
      sudo cp /etc/default/grub "/etc/default/grub.bak-$(date +%Y%m%d-%H%M%S)"
      sudo sed -i -E "s/^(GRUB_CMDLINE_LINUX_DEFAULT=\"[^\"]*)\"/\1 gpu_unlock.targets=${bus}\"/" /etc/default/grub
    fi
    grep '^GRUB_CMDLINE_LINUX_DEFAULT' /etc/default/grub
    sudo update-grub
    echo ">>> Reboot now. If a blue 'MOK management' screen appears: Enroll MOK -> Continue -> password."
    echo ">>> After the reboot check with: ./run.sh status   (gpu_unlock state must be 'unlocked')"
    ;;

  remove-unlock)
    oplog remove-unlock
    sudo cp /etc/default/grub "/etc/default/grub.bak-$(date +%Y%m%d-%H%M%S)"
    sudo sed -i -E 's/ ?gpu_unlock\.targets=[^" ]*//' /etc/default/grub
    grep '^GRUB_CMDLINE_LINUX_DEFAULT' /etc/default/grub
    sudo update-grub
    echo ">>> Reboot to boot the stock card again."
    ;;

  *)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
