#!/bin/bash
# Prepared only. Execute once, after explicit build authorization.
set -eo pipefail
base=HARDWARE_ROOT
repo="$base/preparation_2022_2/kria-vitis-platforms"
ip="$base/ip_official_vai3_0/DPUCZDX8G_ip_repo_VAI_v3.0"
audit="$base/official_reference_audit"
check_ram() {
  awk '/^MemAvailable:/ {ok=1; if ($2 < 25*1024*1024) exit 1} END {if (!ok) exit 1}' /proc/meminfo || {
    echo 'STOP: MemAvailable below 25 GiB'; exit 1;
  }
}
check_ram
# Atomic one-shot guard; never resumes or overwrites a previous attempt.
mkdir "$audit"
source TOOLS_ROOT/Xilinx/Vitis/2022.2/settings64.sh
unset TRD_PATH DIR_PATH DIR_PRJ
export DPU_IP_ROOT="$ip"
{
  date -u
  free -h
  vivado -version
  v++ --version
  git -C "$repo" rev-parse HEAD
} > "$audit/environment.txt" 2>&1
git -C "$repo" diff > "$audit/applied.patch"
cp "$base/preparation_2022_2/repositories.lock.json" "$audit/"
cp "$base/downloads/DPUCZDX8G_ip_repo_VAI_v3.0.manifest.json" "$audit/"
cp "$base/scripts/report_routed.tcl" "$audit/"
cp "${BASH_SOURCE[0]}" "$audit/"
python3 - "$repo" "$ip" "$audit/input_sha256.txt" <<'PYHASH'
import sys,hashlib
from pathlib import Path
with open(sys.argv[3],'w') as out:
 for root in map(Path,sys.argv[1:3]):
  for p in sorted(root.rglob('*')):
   if p.is_file() and not p.is_symlink() and '.git' not in p.parts and '.Xil' not in p.parts:
    out.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p)+'\n')
PYHASH
cd "$repo/kv260"
make -j1 platform PFM=kv260_ispMipiRx_vcu_DP JOBS=1 2>&1 | tee "$audit/platform.log"
check_ram
make -j1 overlay OVERLAY=benchmark JOBS=1 DPU_IP_ROOT="$ip" \
  VPP='v++ --vivado.synth.jobs 1 --vivado.impl.jobs 1 --vivado.param general.maxThreads=1' \
  2>&1 | tee "$audit/overlay.log"
printf '%s\n' 'Official commands completed; timing, artifacts and final reports still require validation before freezing.' > "$audit/BUILD_COMMANDS_COMPLETED.txt"
# No cleanup, no reset, no timeout/kill, no deployment, no rerun on failure.
