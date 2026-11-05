# VascX workspace (third-party models)

The experimental side for VascX: fetching weights, notebooks, evaluation, prototyping. The
deployable, tidy side is [`maps/fundus_vessels`](../maps/fundus_vessels/), which packages the
weights from here. Same split as `AIREADI_Laterality_App/` → `maps/aireadi_laterality/`.

We did not train these models.

| File | What |
|---|---|
| `vessels_walkthrough.ipynb` | photos from Orthanc → mask step by step; **paper model vs new model** with overlays; inside the model; vessel density |
| `compare_paper_model.py` | the paper-vs-new check as a script: `--patient` / `--study` from Orthanc → table + `results/paper_vs_new_<date>.csv` |
| `vessels.py` | prototype inference on an RGB array: crop → model (+ flips) → mask. Promoted to `maps/fundus_vessels/model_spec.py` |
| `orthanc.py` | fetch a patient or study from Orthanc into a temporary folder (`$ORTHANC_URL`) |
| `fetch_model.sh` | weights from a pinned Hugging Face commit, sha256-checked: `models/` (new, packaged into the MAP), `reference_models/` (paper's) |
| `requirements.txt`, `setup_env.sh` | the `vascx` pyenv env; the top block of requirements matches the MAP image |

**The check that decides the model.** The paper's model (`vessels_july24`, the 5-model ensemble evaluated in
Vargas Quiros et al., *TVST* 2025, [doi:10.1167/tvst.14.7.19](https://doi.org/10.1167/tvst.14.7.19)) is run with the
authors' own code, exactly as their published notebooks do, and compared with the new model (`vessels_may26`) run
with ours, on the same photos. Different models, so "roughly match" is the bar. The new model is also run with
the authors' code, which must equal ours (Dice ~0.999). First result, 2026-10-30, 6 Maestro2 photos in our
Orthanc: paper vs new Dice 0.927–0.937, vessel coverage 7.6–8.1 % vs 7.2–7.9 %; authors' code vs ours 0.9995.
Earlier trial on 9 AI-READI photos (Maestro2, Triton, Aurora, Eidon): 0.80–0.94, lowest on Eidon.

Data always comes from Orthanc: load your own first (`MONAI_Deploy_Express/LOCAL_SCRIPTS/upload.sh /path/to/dicom`).

Nothing here imports from `maps/`. The folder is self-contained, so the notebook keeps working if it moves.

| | |
|---|---|
| Model | VascX vessel segmentation, `vessels/vessels_may26.pt` (TorchScript) |
| Source | [huggingface.co/Eyened/vascx](https://huggingface.co/Eyened/vascx), commit `962c83a`; code [github.com/eyened/retinalysis-vascx](https://github.com/eyened/retinalysis-vascx) |
| Paper | Vargas Quiros et al., *VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images*, TVST 2025, [doi:10.1167/tvst.14.7.19](https://doi.org/10.1167/tvst.14.7.19) (preprint [arXiv:2409.16016](https://arxiv.org/abs/2409.16016)). Evaluates the older `vessels_july24` |
| Licence | **AGPL-3.0** (weights and the `retinalysis-*` code). Fine for internal research; check before distributing. |
| Architecture | One ConvNeXt-V2-tiny + UperNet, RGB in, 2 classes (background, vessel) |
| Trained on | CHASEDB, DRHAGIS, FIVES, HRF, LEUVEN, RETA, RS (from the config embedded in the file) |
| Input | Colour fundus photo, cropped to the fundus circle and resized to 1024×1024, ImageNet-normalised |

## Fetch

```bash
bash VascX_App/fetch_model.sh     # -> VascX_App/models/vessels/vessels_may26.pt, sha256-checked
```

`models/` is gitignored. Package the MAP with `-m VascX_App/models`.

## Reading the embedded config

The training and inference config travels inside the `.pt` file:

```python
import json, torch
extra = {"config.yaml": ""}
torch.jit.load("VascX_App/models/vessels/vessels_may26.pt", map_location="cpu", _extra_files=extra)
print(json.loads(extra["config.yaml"])["inference"])
```
