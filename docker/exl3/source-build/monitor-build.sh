#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=${1:?run root required}
DONE_MARKER=${DONE_MARKER:-s1.exit}
printf 'timestamp\tmem_available_kib\tdisk_available_bytes\n' >> "$RUN_ROOT/resources.tsv"
while [[ ! -f "$RUN_ROOT/$DONE_MARKER" ]]; do
    printf '%s\t%s\t%s\n' "$(date -Is)" \
        "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" \
        "$(df -B1 --output=avail "$RUN_ROOT" | tail -1 | tr -d ' ')" \
        >> "$RUN_ROOT/resources.tsv"
    sleep 15
done
