#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=${RUN_ROOT:?new absolute stage directory required}
REFERENCE_ROOT=${REFERENCE_ROOT:?successful S1 reference required}
BUILD_STAGE=${BUILD_STAGE:?s2 or s3 required}
[[ $BUILD_STAGE == s2 || $BUILD_STAGE == s3 ]]
[[ $RUN_ROOT == /* && $RUN_ROOT != "$REFERENCE_ROOT" ]]
[[ ! -e $RUN_ROOT/stage.exit && ! -e $RUN_ROOT/s1.exit ]]
[[ $(cat "$REFERENCE_ROOT/s1-reference.exit") == 0 ]]
reference_id=$(jq -er '.[0].Id' "$REFERENCE_ROOT/manifests/s1-full-image.json")
jq -e --arg image "$reference_id" \
    '.status == "passed" and .imageId == $image and .usage.completion_tokens > 0' \
    "$REFERENCE_ROOT/manifests/model-api.json" >/dev/null
if [[ $BUILD_STAGE == s3 ]]; then
    PRIOR_STAGE_ROOT=${PRIOR_STAGE_ROOT:?S3 requires the validated S2 run}
    [[ $(cat "$PRIOR_STAGE_ROOT/model-validation.exit") == 0 ]]
    prior_id=$(jq -er '.[0].Id' "$PRIOR_STAGE_ROOT/manifests/final-image.json")
    jq -e --arg image "$prior_id" --arg reference "$reference_id" \
        '.imageId == $image and .referenceImageId == $reference
        and .threePercentGatePassed == true' \
        "$PRIOR_STAGE_ROOT/manifests/model-benchmark.json" >/dev/null
    jq -e --arg image "$prior_id" \
        '.imageId == $image and .status == "passed"' \
        "$PRIOR_STAGE_ROOT/manifests/model-api.json" >/dev/null
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/manifests"
export RUN_ROOT REFERENCE_ROOT BUILD_STAGE
export EXPORT_OCI=1
trap 'result=$?; printf "%s\n" "$result" > "$RUN_ROOT/stage.exit"' EXIT
DONE_MARKER=stage.exit bash docker/exl3/source-build/monitor-build.sh "$RUN_ROOT" &
bash docker/exl3/source-build/build-official-source.sh
bash docker/exl3/source-build/build-final-reuse.sh
printf 'SLIM_STAGE_BUILD_PASSED=%s\n' "$BUILD_STAGE"
