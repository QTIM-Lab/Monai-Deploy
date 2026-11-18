# Disc detection App

The detection example, alongside laterality (classification) and VascX (segmentation). It trains a box
detector for the optic disc (and fovea) on AI-READI colour photos, using labels made by VascX:
disc mask → disc box, fovea point → fovea box. VascX itself has no box detector.

This folder holds the code and decides what goes into the data. The data itself (cohort, masks, boxes,
checkpoints) lives in the data area, `/scratch90/QTIM/Active/Non-IRB/VascX_disc_detection_and_segmentation/`
(backup on `/persist/QTIM/Active/Non-IRB/`), which has its own README.

| File | What |
|---|---|
| `build_cohort.py` | picks the images from the AI-READI manifest → `cohort.csv` and `splits.csv`, written to the data area and to `dataset/` |
| `fetch_model.sh` | VascX disc and fovea models, from the same pinned commit as `VascX_App/`, sha256-checked → `models/` |
| `make_labels.py` | runs VascX disc and fovea on the cohort → `disc_masks/` and `boxes.csv` in the data area |
| `make_quality.py` | runs VascX's image-quality model (EyePACS-trained: good / usable / bad) → `quality.csv` in the data area |
| `prepare_cache.py` | decodes and crops every photo once (512 × 512 fundusprep crop) → `cache_512/` in the data area, so training never touches DICOMs |
| `train.py` | trains a disc + fovea detector: `--arch fasterrcnn` or `--arch retinanet` (torchvision, same script) → a run folder in the data area's `Models/` |
| `evaluate.py` | re-runs each run's `best.pt` on every test photo (filtered-out ones too) → per-photo boxes in the data area's `Stats/<run>/` |
| `explore_labels.ipynb` | the research journal: Part 1 the labels (2026-11-05), Part 2 the detectors (2026-11-06): training curves, per-device results, misses, behaviour outside the filter, speed. Writes slide figures to the data area's `Stats/slides/`. **Clear its outputs before committing** (they show AI-READI photos) |
| `slides/make_deck.py` | builds the talk (`.pptx`, speaker notes) from the notebook's figures → data area's `Stats/slides/`. Needs python-pptx: run with the `corisdb_curation` env |
| `setup_env.sh` | reuses the `vascx` pyenv env from `VascX_App/` |
| `dataset/` | copies of the data area's `cohort.csv` and `splits.csv` (gitignored: AI-READI UIDs) |

## Environment

The `vascx` pyenv env from `VascX_App/` already has everything (torch, pydicom, fundusprep, scipy, opencv).
Nothing to install:

```bash
cd /scratch90/QTIM/Active/Non-IRB/Monai-Deploy/Disc_Detection_App
source setup_env.sh             # = pyenv activate vascx
```

## Make the labels

```bash
bash fetch_model.sh             # once: VascX disc + fovea weights -> models/, checksums must say OK
python build_cohort.py          # done already; re-run only after changing FIELDS
python make_labels.py --per-device 20   # sample first: 20 photos per device/region, ~1 min
python make_labels.py           # then everything: 15,611 photos, about 1 hour on one GPU
python make_quality.py          # quality rating for every photo, about 30 min (same resume rules)
```

`make_labels.py` resumes: images already in `boxes.csv` are skipped, so an interrupted run (or the
sample run) carries on where it stopped. To start over, delete `boxes.csv` and `disc_masks/` in the
data area. Run the full pass in tmux, it's long. `--workers` sets the DICOM-decoding processes
(default 8). `$AIREADI_ROOT` and `$DISC_DATA` override the default paths.

`boxes.csv` columns: pixel boxes in the original photo (x0, y0 inclusive; x1, y1 exclusive).
`disc_*` is the box of the mask's largest part (`disc_parts` > 1 is worth a look). `fovea_x/y` is the
point, `fovea_peak` its heatmap confidence, and `fovea_x0..y1` a square with the disc's diameter as
its side. A non-empty `error` means the photo couldn't be read or cropped.

`quality.csv` columns: `q_good`, `q_usable`, `q_bad` (probabilities) and `quality` (the most likely). "Bad" is
EyeQ's *ungradable for DR grading*, stricter than "disc visible": many hazy photos rated bad still show the disc
clearly. So pick the training cut-off in the notebook's quality section instead of dropping every `bad`.

## Train

```bash
python prepare_cache.py                     # once, ~10 min: 15,611 crops, ~12 GB in the data area's cache_512/
python train.py --arch fasterrcnn --gpu 0   # ~13,950 photos pass the filter; 20 epochs by default
python train.py --arch retinanet  --gpu 1   # same data and metrics, the other detector, in parallel
```

Quick test first: `python train.py --arch fasterrcnn --limit 48 --epochs 1`.

Filter (same for train, val and test): VascX found a disc in one piece, and `q_bad <= 0.99` (`--q-bad-max`).
Above 0.99 label failures jump from under 1 % to 3–20 %; below it 89 % of photos are kept.
The split is AI-READI's, per participant, and filtering never moves anyone between splits.

Each run writes `Models/<arch>_<time>/` in the data area: `best.pt` (best on val), `config.json`, `log.csv`
(loss and val metrics per epoch) and `test.csv` (best model on test, overall and per device).
Metrics, per class (disc, fovea), using the top-scoring box: mean IoU with the VascX box, `hit50` (share with
IoU ≥ 0.5), and `centre_dd`, the median centre error in disc diameters.

MONAI's RetinaNet (`monai.apps.detection`) uses its own box format and training wrapper, so it would be a
separate script, compared on the same cache, split and metrics.

## Steps

1. **Cohort** (done): 15,611 standard-field colour photos from 1,065 participants. Kept: Aurora; Eidon,
   Maestro2 and Triton macula; Triton disc. Left out: Eidon mosaic, nasal and temporal periphery, and
   Maestro2 wide field, because VascX wasn't trained on wide fields.
   The split is AI-READI's own `recommended_split`, per participant.
2. **Labels**: `make_labels.py`, above. Checked on 2 photos per device (2026-11-05): disc and fovea
   boxes land correctly on all of them. Photos with no visible disc get no disc box.
3. **Check labels**: `explore_labels.ipynb`. First look (all 15,611 labelled): anatomy checks out; the failures
   are mostly ungradable photos (dark, washed out), concentrated in Aurora. `make_quality.py` rates every photo;
   the notebook's quality section then picks the cut-off.
4. **Train**: `prepare_cache.py`, then `train.py` for Faster R-CNN and RetinaNet (above). Smoke-tested
   2026-11-05: both train and evaluate end to end (Faster R-CNN disc IoU 0.80 after 2 epochs on 48 photos).
5. **Evaluate** (2026-11-06): `python evaluate.py`, then Part 2 of the notebook. Both detectors reach disc IoU 0.935 and fovea
   IoU 0.84–0.85 against VascX; RetinaNet is 2.2× faster, so it's the one for a MAP. Most large misses (mostly Eidon) are VascX
   label errors. Still to do: a hand-checked sample, and MONAI RetinaNet.

Training data is read from the AI-READI files directly; there are too many images to send through
Orthanc. Demos and the later MAP check still read from Orthanc, as in the other Apps.
