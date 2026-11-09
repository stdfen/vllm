#!/usr/bin/env bash
# Exercise the real entry point without Docker, network, or build side effects.
set -euo pipefail
script=$(realpath "${1:-$(dirname "$0")/build-official-source.sh}")
root=$(mktemp -d)
# Retain the tiny fixture directory for diagnosis; never delete caller paths.
mkdir -p "$root/bin" "$root/repo"
cat > "$root/bin/git" <<'GIT'
#!/usr/bin/env bash
case "$*" in
'rev-parse --show-toplevel') printf '%s\n' "$FIXTURE_ROOT/repo" ;;
'rev-parse HEAD') printf '%040d\n' 1 ;;
'branch --show-current') printf '%s\n' slim ;;
'diff --quiet HEAD --') exit "${DIRTY_TRACKED:-0}" ;;
'ls-files --others --exclude-standard')
    if [[ ${DIRTY_UNTRACKED:-0} == 1 ]]; then echo unexpected.py; fi ;;
*) echo "Unexpected git call: $*" >&2; exit 90 ;;
esac
GIT
cat > "$root/bin/jq" <<'JQ'
#!/usr/bin/env bash
# This is the first operation after the identity gate, not a build success.
echo IDENTITY_GATE_PASSED >&2
exit 87
JQ
chmod +x "$root/bin/git" "$root/bin/jq"
export FIXTURE_ROOT="$root"
export PATH="$root/bin:$PATH"
export RUN_ROOT="$root/run" BUILD_STAGE=s1
export ALLOWED_SOURCE_BRANCH=slim
export EXPECTED_SOURCE_SHA=0000000000000000000000000000000000000001
check() {
    local name=$1 expected=$2 status=0
    shift 2
    env "$@" bash "$script" > "$root/$name.log" 2>&1 || status=$?
    if [[ $expected == pass ]]; then
        [[ $status == 87 ]]
        grep -q IDENTITY_GATE_PASSED "$root/$name.log"
    else
        [[ $status != 0 ]]
        ! grep -q IDENTITY_GATE_PASSED "$root/$name.log"
    fi
    echo "PASS $name"
}
check clean pass
check wrong-sha fail EXPECTED_SOURCE_SHA=0000000000000000000000000000000000000002
check short-sha fail EXPECTED_SOURCE_SHA=0000001
check wrong-branch fail ALLOWED_SOURCE_BRANCH=other
check tracked-change fail DIRTY_TRACKED=1
check untracked-source fail DIRTY_UNTRACKED=1
check missing-sha fail EXPECTED_SOURCE_SHA=
check missing-branch fail ALLOWED_SOURCE_BRANCH=
echo "Evidence: $root"
