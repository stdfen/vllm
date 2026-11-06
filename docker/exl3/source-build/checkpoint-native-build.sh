#!/usr/bin/env bash
# Preserve only this source build's own incremental state before changing jobs.
set -euo pipefail
SOURCE_ROOT=$(git rev-parse --show-toplevel)
SETUP_PID=${SETUP_PID:?source setup.py process required}
NINJA_PID=${NINJA_PID:?its native Ninja process required}
NATIVE_BUILD_INPUT=${NATIVE_BUILD_INPUT:?new absolute checkpoint directory required}
[[ $NATIVE_BUILD_INPUT == /* && $SETUP_PID =~ ^[0-9]+$ && $NINJA_PID =~ ^[0-9]+$ ]]
[[ ! -e $NATIVE_BUILD_INPUT/SHA256SUMS ]]
ps -p "$SETUP_PID" -o args= | grep -q 'setup.py bdist_wheel'
[[ $(ps -p "$NINJA_PID" -o comm= | xargs) == ninja ]]
cmake_pid=$(ps -p "$NINJA_PID" -o ppid= | xargs)
[[ $(ps -p "$cmake_pid" -o ppid= | xargs) == "$SETUP_PID" ]]
git diff --quiet HEAD -- CMakeLists.txt cmake csrc setup.py pyproject.toml
mkdir -p "$NATIVE_BUILD_INPUT"
native_source_files=(CMakeLists.txt cmake csrc setup.py pyproject.toml vllm/envs.py
    vllm/__init__.py tools/setup_deepgemm_pythons.sh tools/build_deepgemm_C.py)
git ls-files -z -- "${native_source_files[@]}" | sort -z | xargs -0 sha256sum \
    > "$NATIVE_BUILD_INPUT/native-sources.sha256"
start_ticks=$(awk '{print $22}' "/proc/$NINJA_PID/stat")
resume_ninja() {
    if [[ -f /proc/$NINJA_PID/stat ]] \
        && [[ $(awk '{print $22}' "/proc/$NINJA_PID/stat") == "$start_ticks" ]]; then
        sudo -n kill -CONT "$NINJA_PID"
    fi
}
trap resume_ninja EXIT HUP INT TERM
sudo -n kill -STOP "$NINJA_PID"
echo 'Ninja scheduler paused; waiting for existing compiler children to finish.'
deadline=$((SECONDS + 300))
while ps --ppid "$NINJA_PID" -o stat= | grep -qv '^[[:space:]]*Z'; do
    (( SECONDS < deadline ))
    sleep 2
done
echo 'Compiler children finished; creating a consistent incremental checkpoint.'
workspace="/proc/$SETUP_PID/root/workspace"
sudo -n bash -c 'cd "$1" && sha256sum -c "$2" >/dev/null' _ \
    "$workspace" "$NATIVE_BUILD_INPUT/native-sources.sha256"
sudo -n tar -C "$workspace" -cf - .deps build csrc | gzip -1 \
    > "$NATIVE_BUILD_INPUT/native-state.tar.gz.part"
mv "$NATIVE_BUILD_INPUT/native-state.tar.gz.part" "$NATIVE_BUILD_INPUT/native-state.tar.gz"
jq -n --arg revision "$(git rev-parse HEAD)" \
    --arg inputs "$(sha256sum "$NATIVE_BUILD_INPUT/native-sources.sha256" | cut -d' ' -f1)" \
    --arg base "$(jq -er .buildBase docker/exl3/source-build/sources.lock.json)" \
    '{schemaVersion:1,sourceRevision:$revision,nativeSourcesSha256:$inputs,buildBase:$base,
      cudaVersion:"13.0.3",nvccThreads:2,torchCudaArchList:"12.0",
      buildType:"RelWithDebInfo",provenance:"Current source build incremental objects and dependency Git checkouts"}' \
    > "$NATIVE_BUILD_INPUT/state.json"
(cd "$NATIVE_BUILD_INPUT" && sha256sum native-state.tar.gz native-sources.sha256 state.json > SHA256SUMS)
echo 'NATIVE_CHECKPOINT_VERIFIED'
