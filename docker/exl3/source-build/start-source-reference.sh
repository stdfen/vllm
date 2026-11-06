#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:?absolute run directory required}
SOURCE_ARCHIVE=${SOURCE_ARCHIVE:?pinned ExLlamaV3 archive required}
EXL3_PLUGIN_WHEEL=${EXL3_PLUGIN_WHEEL:?pinned plugin wheel required}
SOURCE_ROOT=$(git rev-parse --show-toplevel)
[[ $RUN_ROOT == /* && -f $SOURCE_ARCHIVE && -f $EXL3_PLUGIN_WHEEL ]]
[[ ! -e $RUN_ROOT/s1.pid && ! -e $RUN_ROOT/s1.exit ]]
export RUN_ROOT SOURCE_ARCHIVE
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/final-context"
cp "$EXL3_PLUGIN_WHEEL" "$RUN_ROOT/final-context/"
(cd "$RUN_ROOT/final-context" && sha256sum -c \
    "$SOURCE_ROOT/docker/exl3/source-build/SHA256SUMS.plugin")
nohup bash docker/exl3/source-build/monitor-build.sh "$RUN_ROOT" \
    > "$RUN_ROOT/logs/monitor-launch.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/monitor.pid"
nohup bash -c '
    set +e
    bash docker/exl3/source-build/build-official-source.sh
    result=$?
    if [[ ! -f $RUN_ROOT/s1.exit ]]; then
        printf "%s\n" "$result" > "$RUN_ROOT/s1.exit.tmp"
        mv "$RUN_ROOT/s1.exit.tmp" "$RUN_ROOT/s1.exit"
    fi
    exit "$result"
' > "$RUN_ROOT/logs/s1-launch.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/s1.pid"
nohup bash docker/exl3/source-build/finish-source-reference.sh \
    > "$RUN_ROOT/logs/pipeline-launch.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/pipeline.pid"
printf 'S1_PID=%s\nPIPELINE_PID=%s\n' \
    "$(cat "$RUN_ROOT/s1.pid")" "$(cat "$RUN_ROOT/pipeline.pid")"
