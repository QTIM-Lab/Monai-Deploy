# Trains a left/right eye classifier on the cached AI-READI fundus images and writes a
# TorchScript model where `monai-deploy package` expects it.
#
# Run prepare_data.py first. This script never touches a DICOM.

import os

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

CACHE_DIR = os.path.join(os.path.curdir, "cache")
CLASSES = ["L", "R"]

BATCH_SIZE = 32
MAX_EPOCHS = 8
LEARNING_RATE = 1e-4

# ImageNet statistics, because the backbone is pretrained on it.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_split(split):
    """Load a cached split as NCHW float tensors, normalised the way the backbone expects."""
    x = np.load(os.path.join(CACHE_DIR, f"{split}_x.npy"))  # (N, 224, 224, 3) uint8
    y = np.load(os.path.join(CACHE_DIR, f"{split}_y.npy"))

    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)  # (N, 3, 224, 224)
    x = (x - MEAN) / STD
    return TensorDataset(x, torch.from_numpy(y))


def augment(x):
    """Light augmentation applied on-GPU per batch.

    Deliberately no horizontal flip. Mirroring a fundus image turns a left eye into a
    right eye, so it would invert the label and cap accuracy at chance.
    """
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])  # vertical only
    return x


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    correct = total = 0
    for x, y in loader:
        pred = net(x.to(device)).argmax(dim=1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
    return correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader = DataLoader(load_split("train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(load_split("val"), batch_size=BATCH_SIZE)
    test_loader = DataLoader(load_split("test"), batch_size=BATCH_SIZE)

    # ResNet18 rather than the DenseNet121 both MedNIST tutorials used, to show the MAP
    # does not care what produced the weights.
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Linear(net.fc.in_features, len(CLASSES))
    net = net.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
        net.train()
        running = 0.0
        for x, y in train_loader:
            x, y = augment(x.to(device)), y.to(device)
            opt.zero_grad()
            loss = loss_function(net(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)

        val_acc = evaluate(net, val_loader, device)
        print(f"epoch {epoch + 1}/{MAX_EPOCHS}  loss {running / len(train_loader.dataset):.4f}  val_acc {val_acc:.4f}")

    print(f"\ntest accuracy: {evaluate(net, test_loader, device):.4f}")

    # - Save the TorchScript model - #
    models_root = os.environ.get("HOLOSCAN_MODEL_PATH")
    models_root = models_root if models_root else os.path.join(os.path.curdir, "models")

    # The "model" subfolder is the model's *name*. The SDK's NamedModel requires every child
    # of the models folder to be a directory, so models/model.ts on its own is not detected.
    model_dir = os.path.join(models_root, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "model.ts")

    torch.jit.script(net.cpu().eval()).save(model_file)
    print(f"Saved TorchScript model to: {model_file}")
