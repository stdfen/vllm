#!/usr/bin/env bash
set -euo pipefail

container=${1:?usage: watch_memory.sh CONTAINER OUTPUT [INTERVAL]}
output=${2:?usage: watch_memory.sh CONTAINER OUTPUT [INTERVAL]}
interval=${3:-5}

printf 'timestamp\tcontainer_pid\tmem_available_kib\tmem_free_kib\tcached_kib\trss_kib\tpss_kib\tprivate_dirty_kib\tgpu_used_mib\tgpu_free_mib\n' >"${output}"
while [[ "$(docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null || true)" == "true" ]]; do
  pid=$(docker inspect -f '{{.State.Pid}}' "${container}")
  read -r mem_available mem_free cached < <(
    awk '/^MemAvailable:/{a=$2}/^MemFree:/{f=$2}/^Cached:/{c=$2}END{print a,f,c}' /proc/meminfo
  )
  read -r rss pss private_dirty < <(
    awk '/^Rss:/{r=$2}/^Pss:/{p=$2}/^Private_Dirty:/{d=$2}END{print r+0,p+0,d+0}' "/proc/${pid}/smaps_rollup"
  )
  IFS=, read -r gpu_used gpu_free < <(
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' '
  )
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${pid}" "${mem_available}" "${mem_free}" "${cached}" \
    "${rss}" "${pss}" "${private_dirty}" "${gpu_used}" "${gpu_free}" >>"${output}"
  sleep "${interval}"
done
