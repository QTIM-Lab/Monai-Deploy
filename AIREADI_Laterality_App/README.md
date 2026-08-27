# AI-READI Laterality Classifier App

# Classifies a retinal fundus DICOM as left (L) or right (R) eye.
# Same three-operator shape as the MedNIST apps, with a DICOM loader instead of PIL
# and a ResNet18 instead of DenseNet121.

```bash
pip install -r AIREADI_Laterality_App/requirements-train.txt
pip install torchvision --index-url https://download.pytorch.org/whl/cu130
```

```bash
export AIREADI_ROOT="/persist/public_datasets/ai_readi"
export HOLOSCAN_INPUT_PATH="$PWD/AIREADI_Laterality_App/demo_input"
export HOLOSCAN_OUTPUT_PATH="$PWD/AIREADI_Laterality_App/output"
export HOLOSCAN_MODEL_PATH="$PWD/AIREADI_Laterality_App/models"

echo $AIREADI_ROOT
echo $HOLOSCAN_INPUT_PATH
echo $HOLOSCAN_OUTPUT_PATH
echo $HOLOSCAN_MODEL_PATH
```

# Prepare data -> writes cache/*.npy and demo_input/*.dcm
# Filters the manifest to Topcon Maestro2 macula-centred colour photography (2119 images,
# 1036 patients, uniform 1958x2576x3), splits BY PATIENT, decodes once at ~400 ms/image.
```bash
cd AIREADI_Laterality_App && python prepare_data.py && cd -
```

# Train -> writes models/model/model.ts   (~40 s on one A4000)
```bash
cd AIREADI_Laterality_App && python train.py && cd -

ls -la AIREADI_Laterality_App/models/model/
```

# The "model" subfolder is the model's NAME. Ex: "models/<modelname>/model.ts"

# Package the MAP (-m points at the models folder)
```bash
monai-deploy package AIREADI_Laterality_App \
  -c AIREADI_Laterality_App/app.yaml \
  -m AIREADI_Laterality_App/models \
  -t airead_laterality_app:1.0 \
  --platform x86_64 \
  -l DEBUG
```

```bash
docker images | grep airead_laterality_app
```

# Run the MAP. --uid/--gid must match the packaging user (holoscan 1000:1000),
# otherwise pip's --user site-packages drop off sys.path and imports fail
```bash
monai-deploy run \
  --uid 1000 \
  --gid 1000 \
  -i "$HOLOSCAN_INPUT_PATH" \
  -o "$HOLOSCAN_OUTPUT_PATH" \
  airead_laterality_app-x64-workstation-dgpu-linux-amd64:1.0
```

# Result: L or R in output.json, plus a DICOM SR from DICOMTextSRWriterOperator.
# The demo DICOM filenames carry the ground truth (..._cfp_l_... / ..._cfp_r_...).
```bash
cat AIREADI_Laterality_App/output/output.json
ls -la AIREADI_Laterality_App/output/
```

# Notes
# 1. No horizontal flip augmentation. Mirroring a fundus image turns a left eye into a
#    right eye, so it inverts the label and caps accuracy at chance.
# 2. Splits are by patient, not by image. Each patient contributes both eyes, so an
#    image-level split would put the same person in train and test.
# 3. Preprocessing is duplicated in prepare_data.py/train.py and in the operator's
#    preprocess(). They must agree. Drift there degrades the model silently.
# 4. torchvision is host-only. TorchScript bakes the model source into model.ts, so the
#    MAP does not need the architecture library at inference time.
