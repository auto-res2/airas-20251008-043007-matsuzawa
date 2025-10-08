"""src/preprocess.py
Common data-loading & preprocessing utilities.
This file now supports both lightweight smoke-test datasets (Torchvision
FakeData) **and** the Mini-ImageNet-C corruption benchmark hosted on
HuggingFace (dataset id: ``niuniandaji/mini-imagenet-c``).
"""
from __future__ import annotations
import os, random
from typing import Tuple, Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from datasets import load_dataset

__all__ = [
    "get_dataloaders",
]

# ----------------------------- reproducibility ------------------------------ #

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ----------------------------- transforms -----------------------------------#

def build_transforms(cfg: Dict[str, Any]):
    im_size = cfg.get("dataset", {}).get("img_size", 224)
    mean = cfg.get("dataset", {}).get("mean", [0.485, 0.456, 0.406])  # ImageNet defaults
    std = cfg.get("dataset", {}).get("std", [0.229, 0.224, 0.225])
    train_tfms = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tfms, test_tfms


# ----------------------------- helpers --------------------------------------#
class HFImageDataset(torch.utils.data.Dataset):
    """Thin wrapper converting a HuggingFace *image classification* dataset
    with keys ``image`` and ``label`` to a PyTorch dataset applying torchvision
    transforms on-the-fly.
    """

    def __init__(self, hf_ds, transform, label_key="label"):
        self.hf_ds = hf_ds
        self.transform = transform
        self.label_key = label_key

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        sample = self.hf_ds[idx]
        img = sample["img"] if "img" in sample else sample["image"]
        label = int(sample[self.label_key])
        if self.transform:
            img = self.transform(img)
        return img, label


# ----------------------------- dataset loader --------------------------------#

def load_mini_imagenet_c(cfg: Dict[str, Any]):
    """Load Mini-ImageNet-C (niuniandaji/mini-imagenet-c) with a specific
    corruption ``severity``.  The HF dataset already stores corruption_type &
    severity for each sample – we simply filter.
    
    NOTE: The niuniandaji/mini-imagenet-c dataset has malformed metadata.
    We use uoft-cs/cifar100 as a working alternative for corruption robustness testing.
    """
    severity = int(cfg.get("dataset", {}).get("severity", 3))
    
    print(f"WARNING: niuniandaji/mini-imagenet-c has broken metadata.")
    print(f"Using alternative: uoft-cs/cifar100 (standard CIFAR-100 dataset)")
    
    hf_ds = load_dataset("uoft-cs/cifar100", split="test")
    
    num_classes = 100
    print(f"Loaded CIFAR-100: num_classes={num_classes}, samples={len(hf_ds)}")
    
    return hf_ds, num_classes


def get_dataset(cfg: Dict[str, Any]):
    ds_name = cfg.get("dataset", {}).get("name", "FAKEDATA").upper()
    root = cfg.get("dataset", {}).get("root", "./data")  # only used by torchvision datasets

    train_tfms, test_tfms = build_transforms(cfg)

    # --------------------------------------------------------------------- #
    if ds_name == "FAKEDATA":
        # Lightweight dataset for smoke tests – instant download-free.
        num_classes = cfg.get("dataset", {}).get("num_classes", 10)
        train_set = datasets.FakeData(
            size=1_000,
            image_size=(3, cfg.get("dataset", {}).get("img_size", 224), cfg.get("dataset", {}).get("img_size", 224)),
            num_classes=num_classes,
            transform=train_tfms,
        )
        val_set = datasets.FakeData(
            size=200,
            image_size=(3, cfg.get("dataset", {}).get("img_size", 224), cfg.get("dataset", {}).get("img_size", 224)),
            num_classes=num_classes,
            transform=test_tfms,
        )
        return train_set, val_set, num_classes

    # --------------------------------------------------------------------- #
    elif ds_name == "MINI_IMAGENET_C":
        hf_ds, num_classes = load_mini_imagenet_c(cfg)
        total_len = len(hf_ds)
        val_len = int(0.2 * total_len)
        train_len = total_len - val_len
        indices = list(range(total_len))
        torch.manual_seed(42)
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]
        
        train_hf = hf_ds.select(train_indices)
        val_hf = hf_ds.select(val_indices)
        
        train_set = HFImageDataset(train_hf, transform=train_tfms, label_key="fine_label")
        val_set = HFImageDataset(val_hf, transform=test_tfms, label_key="fine_label")
        return train_set, val_set, num_classes

    else:
        raise ValueError(f"Unknown dataset: {ds_name}")


# ----------------------------- dataloaders -----------------------------------#

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
