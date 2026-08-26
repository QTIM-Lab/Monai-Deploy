# Simple_Image_Processing

```bash
mkdir -p Simple_Image_Processing/test_input_folder
wget \
  "https://user-images.githubusercontent.com/1928522/133383228-2357d62d-316c-46ad-af8a-359b56f25c87.png" \
  -O Simple_Image_Processing/test_input_folder/normal-brain-mri-4.png
```

```bash
export HOLOSCAN_INPUT_FOLDER="$PWD/Simple_Image_Processing/test_input_folder"
export HOLOSCAN_INPUT_PATH="$PWD/Simple_Image_Processing/test_input_folder/normal-brain-mri-4.png"
export HOLOSCAN_OUTPUT_PATH="$PWD/Simple_Image_Processing/output"

echo $HOLOSCAN_INPUT_FOLDER
echo $HOLOSCAN_INPUT_PATH
echo $HOLOSCAN_OUTPUT_PATH
```

```bash
monai-deploy package Simple_Image_Processing -c Simple_Image_Processing/app.yaml -t simple_imaging_app:1.0 --platform x86_64 -l DEBUG
```

# Fails as user = 1000:1000 (holoscan: /home/holoscan/.local/...) "built" this above (you can't see but it's part of monai deploy)
```bash
monai-deploy run -i Simple_Image_Processing/test_input_folder -o Simple_Image_Processing/output_path simple_imaging_app-x64-workstation-dgpu-linux-amd64:1.0
```

# Works as you are a root docker user...not ideal
```bash
docker run --rm \
  --gpus all \
  --ulimit stack=33554432 \
  -v "$(pwd)/Simple_Image_Processing/test_input_folder:/var/holoscan/input" \
  -v "$(pwd)/Simple_Image_Processing/output_path:/var/holoscan/output" \
  simple_imaging_app-x64-workstation-dgpu-linux-amd64:1.0
```

# Works as dsigned if you supply building users uid\gid
```bash
monai-deploy run \
  --uid 1000 \
  --gid 1000 \
  -i Simple_Image_Processing/test_input_folder \
  -o Simple_Image_Processing/output_path \
  simple_imaging_app-x64-workstation-dgpu-linux-amd64:1.0
```


# Not tested to completetion due to package and cuda compatibility issues...might not be possible
```bash
python Simple_Image_Processing/app.py \
  --input_folder $HOLOSCAN_INPUT_FOLDER \
  --output_folder $HOLOSCAN_OUTPUT_PATH
```

