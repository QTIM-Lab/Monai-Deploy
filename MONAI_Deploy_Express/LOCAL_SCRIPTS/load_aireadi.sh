#!/usr/bin/env bash
# Load AI-READI participants' colour fundus photos (all cameras) into the practice Orthanc: a reproducible test state.
#
#   LOCAL_SCRIPTS/load_aireadi.sh 7034 1019 7115            # add them (already-stored files are skipped)
#   LOCAL_SCRIPTS/load_aireadi.sh 7034 --replace            # delete each patient first, then upload (re-triggers auto-run)
#
# QA baseline: LOCAL_SCRIPTS/reset.sh orthanc, then this script with the test patients (LOCAL_PATCHES/TESTING.md).
# AIREADI_ROOT (env, or .env) = your AI-READI copy, the folder holding retinal_photography/.
# With auto-run on, prefix AUTORUN_OK=1 (each new series then starts runs immediately: that is the point of a test).
set -euo pipefail
cd "$(dirname "$0")/.."
AIREADI_ROOT=${AIREADI_ROOT:-$(sed -n 's/^AIREADI_ROOT=//p' .env)}
[ -d "${AIREADI_ROOT:-}/retinal_photography/cfp" ] || { echo "set AIREADI_ROOT (env or .env) to your AI-READI copy" >&2; exit 1; }

REPLACE=0; PATIENTS=()
for a in "$@"; do [ "$a" = "--replace" ] && REPLACE=1 || PATIENTS+=("$a"); done
[ ${#PATIENTS[@]} -gt 0 ] || { echo "usage: $0 <participant id>... [--replace]" >&2; exit 1; }

for p in "${PATIENTS[@]}"; do
  [ $REPLACE = 1 ] && python3 LOCAL_SCRIPTS/results.py "$p" --delete >/dev/null && echo "$p: deleted from Orthanc"
  found=0
  for dir in "$AIREADI_ROOT"/retinal_photography/cfp/*/"$p"; do
    [ -d "$dir" ] || continue
    found=1
    echo "$p: $(basename "$(dirname "$dir")")"
    LOCAL_SCRIPTS/upload.sh "$dir" | grep -v "^Orthanc now"
  done
  [ $found = 1 ] || echo "$p: no colour photos under $AIREADI_ROOT/retinal_photography/cfp/*/$p" >&2
done
python3 LOCAL_SCRIPTS/results.py "${PATIENTS[@]}"
