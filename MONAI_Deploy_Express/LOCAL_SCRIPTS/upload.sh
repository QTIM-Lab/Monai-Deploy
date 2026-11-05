#!/usr/bin/env bash
# Upload every DICOM file under a folder to the practice Orthanc. Safe to re-run: Orthanc keeps one copy
# per SOPInstanceUID and reports repeats as AlreadyStored. Non-DICOM files are rejected and counted.
#
#   LOCAL_SCRIPTS/upload.sh /path/to/dicom/cache [parallel=4]
#
# Refuses while auto-run is on (each new series would start a run at once, in parallel). Auto-run is off by
# default (.env); if you turned it on, turn it off again with:
#   docker compose up -d orthanc
set -euo pipefail
DIR=${1:?usage: upload.sh <dicom folder> [parallel]}
PARALLEL=${2:-4}
ORTHANC=http://127.0.0.1:18042

if [ "$(docker exec mdl-orthanc printenv AUTOROUTE 2>/dev/null)" != "off" ] && [ "${AUTORUN_OK:-}" != 1 ]; then
  echo "Auto-run is ON: every new series would start its own run at once, in parallel." >&2
  echo "  Turn it off (the .env default):  docker compose up -d orthanc" >&2
  echo "  To upload with auto-run anyway: AUTORUN_OK=1 $0 $*" >&2
  exit 1
fi

find "$DIR" -type f -print0 |
  xargs -0 -P "$PARALLEL" -I{} sh -c \
    'curl -s -X POST '"$ORTHANC"'/instances --data-binary @"$1" | python3 -c "import sys,json
try: r=json.load(sys.stdin); print(r.get(\"Status\") or \"Rejected (not DICOM)\")
except Exception: print(\"Rejected (not DICOM)\")"' _ {} |
  sort | uniq -c | sed 's/^/  /'
curl -s $ORTHANC/statistics | python3 -c "import sys,json; s=json.load(sys.stdin); print(f\"Orthanc now: {s['CountPatients']} patients, {s['CountStudies']} studies, {s['CountSeries']} series, {s['CountInstances']} instances\")"
