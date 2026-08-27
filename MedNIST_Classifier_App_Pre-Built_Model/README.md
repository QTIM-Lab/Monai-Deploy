# MedNIST Classifier App (Pre-Built Model)

# Same app as MedNIST_Classifier_App, but the model is downloaded instead of trained.
# There is no all_in_one_train.py here and no requirements-train.txt.

```bash
pip install -r MedNIST_Classifier_App_Pre-Built_Model/requirements.txt
pip install gdown
```

# Download the pre-built model + test images (one zip holds both)
# Unzips to ./input/ (test images) and ./classifier.zip (TorchScript model)
```bash
cd MedNIST_Classifier_App_Pre-Built_Model
gdown "https://drive.google.com/uc?id=1IoEJZFFixcNtPPKeKZfD_xSJSFQCbawl"
unzip -o mednist_classifier_data.zip
cd - # need to be here: ./Monai-Deploy
```

# Place the model where the MAP expects it. "model" is the model's NAME.
# classifier.zip and model.ts are the same format - TorchScript .save() writes a zip
# archive, and the SDK detects it by opening the file, not by extension.
```bash
mkdir -p MedNIST_Classifier_App_Pre-Built_Model/models/model
cp MedNIST_Classifier_App_Pre-Built_Model/classifier.zip \
   MedNIST_Classifier_App_Pre-Built_Model/models/model/

ls -la MedNIST_Classifier_App_Pre-Built_Model/models/model/
```

```bash
export HOLOSCAN_INPUT_PATH="$PWD/MedNIST_Classifier_App_Pre-Built_Model/input"
export HOLOSCAN_OUTPUT_PATH="$PWD/MedNIST_Classifier_App_Pre-Built_Model/output"
export HOLOSCAN_MODEL_PATH="$PWD/MedNIST_Classifier_App_Pre-Built_Model/models"

echo $HOLOSCAN_INPUT_PATH
echo $HOLOSCAN_OUTPUT_PATH
echo $HOLOSCAN_MODEL_PATH
```

# Package the MAP (-m points at the models folder)
```bash
monai-deploy package MedNIST_Classifier_App_Pre-Built_Model \
  -c MedNIST_Classifier_App_Pre-Built_Model/app.yaml \
  -m MedNIST_Classifier_App_Pre-Built_Model/models \
  -t mednist_prebuilt_app:1.0 \
  --platform x86_64 \
  -l DEBUG
```

```bash
docker images | grep mednist_prebuilt_app
```

# Run the MAP. --uid/--gid must match the packaging user (holoscan 1000:1000),
# otherwise pip's --user site-packages drop off sys.path and imports fail.
# The tutorial omits these flags, so its run command fails here.
```bash
monai-deploy run \
  --uid 1000 \
  --gid 1000 \
  -i "$HOLOSCAN_INPUT_PATH" \
  -o "$HOLOSCAN_OUTPUT_PATH" \
  mednist_prebuilt_app-x64-workstation-dgpu-linux-amd64:1.0
```

# Result: the predicted class in output.json, plus a DICOM SR from DICOMTextSRWriterOperator
```bash
cat MedNIST_Classifier_App_Pre-Built_Model/output/output.json
ls -la MedNIST_Classifier_App_Pre-Built_Model/output/
```

# Differences from MedNIST_Classifier_App
# 1. No training. classifier.zip arrives already trained by someone else.
# 2. app.py uses Application.init_app_context(self.argv) instead of AppContext({}),
#    so -i/-o/-m on the command line override the HOLOSCAN_* env vars.
# 3. The operators are byte-identical. Only where the weights come from changed.

# The seam: MEDNIST_CLASSES and the transform in mednist_classifier_operator.py are
# assumptions about a model this app did not train. Nothing in classifier.zip declares
# its class order or its preprocessing. If they disagree, you get a wrong label with no
# error - just a confident answer written into a DICOM SR.
