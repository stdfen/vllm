#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=${RUN_ROOT:?absolute run directory required}
SOURCE_ARCHIVE=${SOURCE_ARCHIVE:?pinned source archive required}
export RUN_ROOT SOURCE_ARCHIVE
mkdir -p "$RUN_ROOT/logs"
exec > >(tee -a "$RUN_ROOT/logs/s1-reference-pipeline.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$RUN_ROOT/s1-reference.exit"' EXIT
while [[ ! -f "$RUN_ROOT/s1.exit" ]]; do
    sleep 15
done
[[ $(cat "$RUN_ROOT/s1.exit") == 0 ]]
[[ -f "$RUN_ROOT/official.oci/index.json" ]]
DONE_MARKER=s1-reference.exit bash docker/exl3/source-build/monitor-build.sh "$RUN_ROOT" &
bash docker/exl3/source-build/build-final-full.sh
printf 'REFERENCE_BUILD_PASSED\n'
