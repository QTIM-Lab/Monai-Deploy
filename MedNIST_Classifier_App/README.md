# MedNIST Classifier App

```bash
pip install -r MedNIST_Classifier_App/requirements-train.txt
mkdir -p MedNIST_Classifier_App/MedNIST_DATA
```

```bash
export MONAI_DATA_DIRECTORY="$PWD/MedNIST_Classifier_App/MedNIST_DATA"
export HOLOSCAN_INPUT_PATH="$PWD/MedNIST_Classifier_App/input"
export HOLOSCAN_OUTPUT_PATH="$PWD/MedNIST_Classifier_App/output"
export HOLOSCAN_MODEL_PATH="$PWD/MedNIST_Classifier_App/models"

echo $MONAI_DATA_DIRECTORY
echo $HOLOSCAN_INPUT_PATH
echo $HOLOSCAN_OUTPUT_PATH
echo $HOLOSCAN_MODEL_PATH
```

# Train -> writes models/model/model.ts
# An empty MedNIST_DATA/MedNIST/ makes the script skip extraction, so clear it first
```bash
python MedNIST_Classifier_App/all_in_one_train.py

ls -la MedNIST_Classifier_App/models/model/
```

# The "model" subfolder is the model's NAME. Ex: "models/<modelname>/model.ts"

# Grab one MedNIST image to classify
```bash
mkdir -p MedNIST_Classifier_App/input MedNIST_Classifier_App/output
cp "$(ls MedNIST_Classifier_App/MedNIST_DATA/MedNIST/AbdomenCT/* | head -1)" \
   MedNIST_Classifier_App/input/
```

# Package the MAP (-m points at the models folder)
```bash
monai-deploy package MedNIST_Classifier_App \
  -c MedNIST_Classifier_App/app.yaml \
  -m MedNIST_Classifier_App/models \
  -t mednist_app:1.0 \
  --platform x86_64 \
  -l DEBUG
```

```bash
docker images | grep mednist_app
```

# Run the MAP. --uid/--gid must match the packaging user (holoscan 1000:1000),
# otherwise pip's --user site-packages drop off sys.path and imports fail
```bash
monai-deploy run \
  --uid 1000 \
  --gid 1000 \
  -i "$HOLOSCAN_INPUT_PATH" \
  -o "$HOLOSCAN_OUTPUT_PATH" \
  mednist_app-x64-workstation-dgpu-linux-amd64:1.0
```

# Result: the predicted class in output.json, plus a DICOM SR from DICOMTextSRWriterOperator
```bash
cat MedNIST_Classifier_App/output/output.json
ls -la MedNIST_Classifier_App/output/
```

# Note: HOLOSCAN_MODEL_PATH is a DIRECTORY here, matching what the MAP entrypoint sets
# (/opt/holoscan/models/). MedNISTClassifierOperator.MODEL_LOCAL_PATH instead treats it as a
# file path, but that fallback only runs outside a MAP, so unset it for a direct python run.
