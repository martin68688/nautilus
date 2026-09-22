#!/bin/bash
set -euo pipefail
# Aborted slow cache copy is retained; this is a FRESH STORE, not verified migration.
test "$(findmnt -n -o UUID /workspace)" = caa1d586-00f0-463a-86c8-6587fcfc8638
test "$(systemctl is-active docker || true)" = inactive
test "$(systemctl is-active containerd || true)" = inactive
test ! -e /workspace/infra/leaf-docker-fresh-v001
mkdir -p /workspace/infra/leaf-docker-fresh-v001/{docker,containerd}
cp -a /etc/fstab /etc/fstab.before-leaf-docker-fresh-20260918
for store in docker containerd; do
  printf '/workspace/infra/leaf-docker-fresh-v001/%s /var/lib/%s none bind,x-systemd.requires-mounts-for=/workspace 0 0\n' "$store" "$store" >> /etc/fstab
  mount --bind /workspace/infra/leaf-docker-fresh-v001/$store /var/lib/$store
done
systemctl daemon-reload
systemctl start containerd.service docker.socket docker.service
docker info --format '{{.ServerVersion}}'
findmnt -n -o SOURCE,TARGET /var/lib/containerd
df -h / /workspace /var/lib/containerd
