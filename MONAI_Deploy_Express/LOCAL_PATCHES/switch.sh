#!/usr/bin/env bash
# Switch Express between the stock and patched gateway / workflow manager, and restart those two.
#
#   LOCAL_PATCHES/switch.sh stock    ghcr IG 0.5.1-3          + ghcr WM 0.2.1-2            (upstream as shipped)
#   LOCAL_PATCHES/switch.sh pr       IG PR #544 (main + fix)  + ghcr WM 0.2.1-2
#   LOCAL_PATCHES/switch.sh fixed    IG PR #544 (main + fix)  + WM fix-parser-race (develop + lock)
#   LOCAL_PATCHES/switch.sh          show the current mode
#
# Edits the two image lines in .env (so a plain `docker compose up` keeps the mode). pr/fixed need build.sh first.
# Restarting the gateway also hands any export it was holding back to RabbitMQ, which redelivers it at once.
set -euo pipefail
cd "$(dirname "$0")/.."
STOCK_IG=ghcr.io/project-monai/monai-deploy-informatics-gateway:0.5.1-3
STOCK_WM=ghcr.io/project-monai/monai-deploy-workflow-manager:0.2.1-2
PR_IG=monai-deploy-informatics-gateway:pr544-af5e15c
FIX_WM=monai-deploy-workflow-manager:fix-parser-race-4e76602

case "${1:-}" in
  stock) IG=$STOCK_IG; WM=$STOCK_WM ;;
  pr)    IG=$PR_IG;    WM=$STOCK_WM ;;
  fixed) IG=$PR_IG;    WM=$FIX_WM ;;
  "")    grep -E '^(INFORMATICS_GATEWAY|WORKFLOW_MANAGER)_IMAGE=' .env; exit 0 ;;
  *)     echo "usage: $0 stock|pr|fixed" >&2; exit 1 ;;
esac
for image in "$IG" "$WM"; do
  docker image inspect "$image" >/dev/null 2>&1 || docker pull -q "$image" >/dev/null 2>&1 || {
    echo "image $image missing: run LOCAL_PATCHES/build.sh" >&2; exit 1; }
done
sed -i -E "s#^INFORMATICS_GATEWAY_IMAGE=.*#INFORMATICS_GATEWAY_IMAGE=$IG#; s#^WORKFLOW_MANAGER_IMAGE=.*#WORKFLOW_MANAGER_IMAGE=$WM#" .env
docker compose up -d informatics-gateway workflow-manager 2>&1 | grep -E "Recreat|Start" | sed 's/^/  /' || true
until curl -sf http://127.0.0.1:5000/health/status >/dev/null && curl -sf "http://127.0.0.1:5001/workflows?pageSize=1" >/dev/null; do sleep 2; done
echo "mode: $1"
docker inspect mdl-ig mdl-wm --format '  {{.Name}}  {{.Config.Image}}'
