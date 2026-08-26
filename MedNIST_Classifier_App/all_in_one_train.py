# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import glob
import PIL.Image
import torch
import numpy as np

from ignite.engine import Events

from monai.apps import download_and_extract
from monai.config import print_config
from monai.networks.nets import DenseNet121
from monai.engines import SupervisedTrainer
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
)
from monai.utils import set_determinism

set_determinism(seed=0)

print_config()



# - Download Dataset - #
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = directory if directory else os.path.join(os.path.curdir, "MedNIST_DATA")
print(root_dir)

# resource = "https://drive.google.com/uc?id=1QsnnkvZyJPcbRoV_ArW8SnE1OTuoVbKE" # Didn't work due to not having file extension in the URL, so using GitHub release instead
resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
md5 = "0bc7306e7427e00ad1c5526a6677552d"

compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
data_dir = os.path.join(root_dir, "MedNIST")
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5)


subdirs = sorted(glob.glob(f"{data_dir}/*/"))

class_names = [os.path.basename(sd[:-1]) for sd in subdirs]
image_files = [glob.glob(f"{sb}/*") for sb in subdirs]

image_files_list = sum(image_files, [])
image_class = sum(([i] * len(f) for i, f in enumerate(image_files)), [])
image_width, image_height = PIL.Image.open(image_files_list[0]).size

print(f"Label names: {class_names}")
print(f"Label counts: {list(map(len, image_files))}")
print(f"Total image count: {len(image_class)}")
print(f"Image dimensions: {image_width} x {image_height}")


# - Setup and Train - #
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim="no_channel"),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),
    ]
)


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_ds = MedNISTDataset(image_files_list, image_class, train_transforms)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DenseNet121(spatial_dims=2, in_channels=1, out_channels=len(class_names)).to(device)
loss_function = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), 1e-5)
max_epochs = 5


def _prepare_batch(batch, device, non_blocking):
    return tuple(b.to(device) for b in batch)


trainer = SupervisedTrainer(device, max_epochs, train_loader, net, opt, loss_function, prepare_batch=_prepare_batch)


@trainer.on(Events.EPOCH_COMPLETED)
def _print_loss(engine):
    print(f"Epoch {engine.state.epoch}/{engine.state.max_epochs} Loss: {engine.state.output[0]['loss']}")


trainer.run()


# - Save the TorchScript model - #
# The tutorial writes "classifier.zip" here and copies it into the models folder as a separate step.
# We write "model.ts" directly into models/model/ instead:
#   * .ts is the conventional extension for a TorchScript archive
#   * models/model/ is the layout `monai-deploy package -m models` expects, where the
#     "model" subfolder name is the model's name
#   * model.ts also matches MedNISTClassifierOperator.MODEL_LOCAL_PATH, so running the
#     app directly (outside a MAP) finds it without any code change
models_root = os.environ.get("HOLOSCAN_MODEL_PATH")
models_root = models_root if models_root else os.path.join(os.path.curdir, "models")
print(models_root)

# The "model" subfolder is the model's *name*. The SDK's NamedModel requires every child of
# the models folder to be a directory, so models/model.ts on its own would not be detected.
model_dir = os.path.join(models_root, "model")
os.makedirs(model_dir, exist_ok=True)
model_file = os.path.join(model_dir, "model.ts")

torch.jit.script(net).save(model_file)
print(f"Saved TorchScript model to: {model_file}")
