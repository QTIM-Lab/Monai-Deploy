#!/usr/bin/env bash
# Download the VascX vessel models and check them.
#
#   bash VascX_App/fetch_model.sh
#
#   models/vessels/vessels_may26.pt            the current model: what the MAP packages (-m VascX_App/models)
#   reference_models/vessels_july24.pt         the paper's model (TVST 2025), for compare_paper_model.py only;
#                                              kept out of models/ so it isn't packaged into the MAP (353 MB)
#
# Pinned to one Hugging Face commit and checksums, so every host gets the same weights.
# To move to a newer release: change REVISION and the file/checksum together, then bump
# ALGORITHM_VERSION in maps/fundus_vessels/model_spec.py.
set -euo pipefail

REPO="Eyened/vascx"
REVISION="962c83a78b8867c2d3028635e7d7d1ee07fff2be"
HERE="$(cd "$(dirname "$0")" && pwd)"

fetch() {  # $1 = file in the HF repo, $2 = sha256, $3 = destination folder
    local out="$3/$(basename "$1")"
    mkdir -p "$3"
    if [ ! -f "$out" ]; then
        curl -fL --retry 3 -o "$out.part" "https://huggingface.co/$REPO/resolve/$REVISION/$1"
        mv "$out.part" "$out"
    fi
    echo "$2  $out" | sha256sum -c -
}

fetch vessels/vessels_may26.pt  9f24c987c020718b344f8c8bd5155be38299cc4e2f323efafa72e6bca48cb052 "$HERE/models/vessels"
fetch vessels/vessels_july24.pt bdabae77502648acd1c176bff6d5e3c8295da60f18f2f84fe1bcd6181d2b2ca4 "$HERE/reference_models"
