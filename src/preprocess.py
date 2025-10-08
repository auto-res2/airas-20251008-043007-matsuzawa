"""src/preprocess.py
Common data-loading & preprocessing utilities.
Handles ImageNet-C and CIFAR-10-C (HuggingFace) in addition to the lightweight
FakeData used by the smoke-tests.
"""
from __future__ import annotations
import os, random
from typing import Tuple, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torch.utils.data import Dataset

# HuggingFace dataset API is optional – import lazily
try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None  #   will raise informative error if invoked


# --------------------------------------------------------------------------- #
# Reproducibility helpers
# --------------------------------------------------------------------------- #

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- #
# Transform builders
# --------------------------------------------------------------------------- #

def _get_stats(ds_name: str):
    """Return dataset-specific channel stats."""
    l = ds_name.lower()
    if "imagenet" in l:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif "cifar" in l:
        return [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    else:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def build_transforms(cfg: Dict[str, Any]):
    ds_name = cfg.get("dataset", {}).get("name", "fakedata")
    im_size = cfg.get("dataset", {}).get("img_size", 224)
    mean = cfg.get("dataset", {}).get("mean")
    std = cfg.get("dataset", {}).get("std")
    if mean is None or std is None:
        mean, std = _get_stats(ds_name)

    test_tfms = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return test_tfms, test_tfms  # train / test identical (no aug) for TTA experiments


# --------------------------------------------------------------------------- #
# HuggingFace wrapper dataset
# --------------------------------------------------------------------------- #
class HFDataset(Dataset):
    """Light wrapper converting a HuggingFace *Split* into Torch Dataset."""

    def __init__(self, hf_ds, transform):
        if load_dataset is None:
            raise RuntimeError("datasets library required but not installed.")
        self.ds = hf_ds
        self.tfm = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]  # ensure int for streaming datasets
        img = item["image"]  # PIL.Image
        label = item["label"]
        img = self.tfm(img)
        return img, label


# --------------------------------------------------------------------------- #
# Dataset dispatcher
# --------------------------------------------------------------------------- #

def get_dataset(cfg: Dict[str, Any]):
    ds_name = cfg.get("dataset", {}).get("name", "fakedata").lower()
    root = cfg.get("dataset", {}).get("root", "./data")

    train_tfms, test_tfms = build_transforms(cfg)

    # --------------------------- FakeData (smoke) --------------------------- #
    if ds_name == "fakedata":
        num_classes = cfg.get("dataset", {}).get("num_classes", 10)
        img_sz = cfg.get("dataset", {}).get("img_size", 64)
        train_set = datasets.FakeData(
            size=1000,
            image_size=(3, img_sz, img_sz),
            num_classes=num_classes,
            transform=train_tfms,
        )
        val_set = datasets.FakeData(
            size=200,
            image_size=(3, img_sz, img_sz),
            num_classes=num_classes,
            transform=test_tfms,
        )
        return train_set, val_set, num_classes

    # --------------------------- ImageNet-C -------------------------------- #
    if "imagenet_c" in ds_name or "imagenet-c" in ds_name:
        if load_dataset is None:
            raise RuntimeError("datasets library required for ImageNet-C loading")
        hf_ds = load_dataset("ang9867/ImageNet-C", split="test", trust_remote_code=False)
        num_classes = 1000
        full = HFDataset(hf_ds, test_tfms)
        # simple random split 90/10 – we do *not* train, but keep API consistent
        val_len = max(1, int(0.1 * len(full)))
        train_len = len(full) - val_len
        train_set, val_set = random_split(full, [train_len, val_len])
        return train_set, val_set, num_classes

    # --------------------------- CIFAR10-C --------------------------------- #
    if "cifar10_c" in ds_name or "cifar10-c" in ds_name:
        if load_dataset is None:
            raise RuntimeError("datasets library required for CIFAR10-C loading")
        hf_ds = load_dataset("randall-lab/cifar10-c", split="test", trust_remote_code=True)
        num_classes = 10
        full = HFDataset(hf_ds, test_tfms)
        val_len = max(1, int(0.1 * len(full)))
        train_len = len(full) - val_len
        train_set, val_set = random_split(full, [train_len, val_len])
        return train_set, val_set, num_classes

    raise ValueError(f"Unknown dataset: {ds_name}")


# --------------------------------------------------------------------------- #
# Dataloader helper
# --------------------------------------------------------------------------- #

def get_dataloaders(cfg: Dict[str, Any], seed: Optional[int] = None):
    if seed is not None:
        seed_everything(seed)

    batch_size = cfg.get("training", {}).get("batch_size", 32)
    num_workers = cfg.get("dataset", {}).get("num_workers", 4)

    train_set, val_set, num_classes = get_dataset(cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes
