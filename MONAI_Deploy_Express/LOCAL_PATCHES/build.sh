#!/usr/bin/env bash
# Build the two fixed MONAI Deploy images used by `switch.sh pr` and `switch.sh fixed`, straight from the fix
# branches at pinned commits. Rebuild on each host; images are not in git.
#
#   informatics-gateway  PR #544 (Project-MONAI, refs/pull/544/head) as is: upstream main + the export de-dup fix
#   workflow-manager     fix-parser-race (our branch: upstream develop + the parser lock + its unit test), from the
#                        QTIM-Lab fork, or from the local clone in ../../upstream/ until that branch is pushed
#
#   LOCAL_PATCHES/build.sh        # ~5 min, then LOCAL_PATCHES/switch.sh pr|fixed
#
# Changed a fix? Commit on its branch (upstream/README.md), update the pinned commit below, rebuild.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CLONES=$(cd "$HERE/../.." && pwd)/upstream
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

IG_URL=https://github.com/Project-MONAI/monai-deploy-informatics-gateway.git
IG_REF=pull/544/head
IG_COMMIT=af5e15cf652f9c1e57cbb7d651885ea484aad38c
IG_TAG=monai-deploy-informatics-gateway:pr544-${IG_COMMIT:0:7}

WM_URL=https://github.com/QTIM-Lab/monai-deploy-workflow-manager.git
WM_REF=fix-parser-race
WM_COMMIT=4e76602c   # git -C upstream/monai-deploy-workflow-manager rev-parse --short fix-parser-race
WM_TAG=monai-deploy-workflow-manager:fix-parser-race-${WM_COMMIT:0:7}

build() {  # name url ref commit tag dockerfile
  local name=$1 url=$2 ref=$3 commit=$4 tag=$5 dockerfile=$6 local_clone="$CLONES/$1" src
  if git -C "$local_clone" cat-file -e "$commit^{commit}" 2>/dev/null; then src=$local_clone; else src=$url; fi
  git init -q "$WORK/$name"
  if [ "$src" = "$local_clone" ]; then  # local clone: the commit is on one of its branches
    git -C "$WORK/$name" fetch -q "$src" 'refs/heads/*:refs/remotes/src/*'
  else                                 # GitHub: fetch the PR / branch ref
    git -C "$WORK/$name" fetch -q "$src" "$ref"
  fi
  git -C "$WORK/$name" -c advice.detachedHead=false checkout -q "$commit"
  echo "== $name @ $(git -C "$WORK/$name" log --oneline -1)  (from $src)"
  docker build -q -f "$WORK/$name/$dockerfile" -t "$tag" "$WORK/$name" >/dev/null
  echo "   built $tag"
}

build monai-deploy-informatics-gateway "$IG_URL" "$IG_REF" "$IG_COMMIT" "$IG_TAG" Dockerfile
build monai-deploy-workflow-manager "$WM_URL" "$WM_REF" "$WM_COMMIT" "$WM_TAG" WorkflowManager.Dockerfile
