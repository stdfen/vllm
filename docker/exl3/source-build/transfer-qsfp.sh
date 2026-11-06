#!/usr/bin/env bash
set -euo pipefail
IMAGE=${IMAGE:?immutable image tag required}
SSH_TARGET=${SSH_TARGET:?QSFP SSH destination required}
QSFP_INTERFACE=${QSFP_INTERFACE:?QSFP interface required}
RUN_ROOT=${RUN_ROOT:?absolute run directory required}
SSH_CIPHER=${SSH_CIPHER:-aes128-gcm@openssh.com}
[[ $IMAGE =~ ^[a-zA-Z0-9./:_-]+$ ]]
[[ $SSH_TARGET =~ ^[a-zA-Z0-9.@_-]+$ ]]
[[ $QSFP_INTERFACE =~ ^[a-zA-Z0-9_-]+$ ]]
[[ $SSH_CIPHER =~ ^[a-zA-Z0-9@._+-]+$ ]]
ssh_args=(-o BatchMode=yes -o Compression=no -c "$SSH_CIPHER")
mkdir -p "$RUN_ROOT/archives" "$RUN_ROOT/manifests"
peer=$(ssh -G "$SSH_TARGET" 2>/dev/null | awk '$1 == "hostname" {print $2; exit}')
ip route get "$peer" | grep -F "dev $QSFP_INTERFACE " >/dev/null
source_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')
image_size=$(docker image inspect "$IMAGE" --format '{{.Size}}')
remote_free=$(ssh "${ssh_args[@]}" "$SSH_TARGET" 'df -B1 --output=avail /var/lib/docker | tail -1')
# Docker load temporarily retains both unpacking input and committed layers.
# The first ARM64 load approached twice the logical image size before cleanup.
(( remote_free >= 2 * image_size + 8 * 1024 * 1024 * 1024 ))
archive="$RUN_ROOT/archives/${source_id#sha256:}.tar.zst"
if [[ ! -f $archive ]]; then
    docker save "$IMAGE" | zstd -T4 -1 -o "$archive.part"
    mv "$archive.part" "$archive"
fi
sha256sum "$archive" > "$archive.sha256"
start=$(date +%s)
# Stream the archive through SSH so the destination does not need space for
# both the compressed archive and Docker's imported layers.
ssh "${ssh_args[@]}" "$SSH_TARGET" "bash -o pipefail -c 'zstd -dc | docker load'" < "$archive"
destination_id=$(ssh "${ssh_args[@]}" "$SSH_TARGET" "docker image inspect '$IMAGE' --format '{{.Id}}'")
[[ $source_id == "$destination_id" ]]
printf 'image=%s\nimage_id=%s\ncompressed_bytes=%s\ntransfer_seconds=%s\ninterface=%s\ncipher=%s\n' \
    "$IMAGE" "$source_id" "$(stat -c %s "$archive")" "$(( $(date +%s) - start ))" \
    "$QSFP_INTERFACE" "$SSH_CIPHER" \
    > "$RUN_ROOT/manifests/qsfp-transfer.txt"
