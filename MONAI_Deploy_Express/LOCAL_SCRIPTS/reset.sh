#!/usr/bin/env bash
# Put the practice stack back to a known state. Every level is safe to re-run.
#
#   LOCAL_SCRIPTS/reset.sh orthanc   delete every patient from the practice Orthanc (stack keeps running)
#   LOCAL_SCRIPTS/reset.sh monai     wipe MONAI state (MongoDB runs/workflows/IG config, MinIO, RabbitMQ,
#                                    task dirs, exited MAP containers), keep Orthanc's images, restart,
#                                    re-register workflows + per-MAP AEs
#   LOCAL_SCRIPTS/reset.sh all       both: factory reset, then re-register
#
# Configs and workflows are reloaded from the files in configs/ and workflows/ (git is the source).
# Auto-run comes back as set in .env (off by default).
set -euo pipefail
cd "$(dirname "$0")/.."
LEVEL=${1:?usage: reset.sh orthanc|monai|all}
ORTHANC=http://127.0.0.1:18042

clear_orthanc() {
  local n=0
  for p in $(curl -s $ORTHANC/patients | python3 -c "import sys,json; print(' '.join(json.load(sys.stdin)))"); do
    curl -s -X DELETE $ORTHANC/patients/$p >/dev/null && n=$((n+1))
  done
  echo "Orthanc: deleted $n patient(s)"
}

wipe() {  # $@ = .md subfolders to empty; their files belong to root (written by containers), so use one
  docker run --rm -v "$PWD/.md:/md" alpine sh -c 'for d in "$@"; do rm -rf /md/$d/* /md/$d/.[!.]* 2>/dev/null; done' _ "$@"
  echo "wiped .md/{$(IFS=,; echo "$*")}"
}

reset_stack() {  # $@ = .md subfolders to wipe
  docker compose down
  wipe "$@"
  docker container prune -f --filter label=org.monai.deploy.app-sdk >/dev/null
  docker compose up -d
  python3 LOCAL_SCRIPTS/register.py
}

case "$LEVEL" in
  orthanc) clear_orthanc ;;
  monai)   reset_stack mongodb minio rabbitmq mdtm mdig ;;
  all)     reset_stack mongodb minio rabbitmq mdtm mdig orthanc ;;
  *)       echo "usage: reset.sh orthanc|monai|all" >&2; exit 2 ;;
esac
