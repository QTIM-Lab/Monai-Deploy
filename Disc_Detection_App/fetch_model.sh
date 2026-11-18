#!/usr/bin/env bash
# Download the VascX disc and fovea models (the label makers) and check them.
#
#   bash Disc_Detection_App/fetch_model.sh
#
#   models/disc/disc_may26.pt      optic disc segmentation -> disc boxes
#   models/fovea/fovea_may26.pt    fovea point (heatmap regression) -> fovea boxes
#   models/quality/quality.pt      image quality (EyePACS-trained, 3 classes) -> drop ungradable photos (855 MB)
#
# Pinned to the same Hugging Face commit as VascX_App/fetch_model.sh, with checksums, so every
# host makes the same labels. Changing either model changes the labels: regenerate the data.
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

fetch disc/disc_may26.pt   8efdc836f6e9b1bbdd084f7c127ca2241c97913a4732f0ff47db9ef8c9e58f4a "$HERE/models/disc"
fetch fovea/fovea_may26.pt 114daf518186122cdbbae66fceeb3fd00f6411b72b99bdc81c1272e36441055a "$HERE/models/fovea"
fetch quality/quality.pt     80034ccb21a57522ba0cb86be0d46d2b659e193b86db909ad6ec2e85e61f87aa "$HERE/models/quality"
