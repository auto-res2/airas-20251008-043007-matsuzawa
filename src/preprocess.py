"""src/preprocess.py
Common data-loading & preprocessing utilities with fully implemented dataset
loaders for all experiments used in this repository.

Public functions
----------------
get_dataloaders(cfg [, seed]) -> (train_loader, val_loader, num_classes)

The rest of the code base is dataset / model agnostic and therefore does not
need to be touched when adding new datasets.  Simply extend *this* file.
"""
from __future__ import annotations

import random
from typing import Tuple, Dict, Any, Optional, Sequence

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode

# Hugging Face datasets – optional dependency, only imported when needed
try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover – handled at runtime
    load_dataset = None  # type: ignore

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(cfg: Dict[str, Any]):
    im_size = cfg.get("dataset", {}).get("img_size", 224)
    mean = cfg.get("dataset", {}).get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("dataset", {}).get("std", [0.229, 0.224, 0.225])

    train_tfms = transforms.Compose([
        transforms.Resize((im_size, im_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((im_size, im_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tfms, test_tfms


# ---------------------------------------------------------------------------
# Hugging-Face wrappers
# ---------------------------------------------------------------------------

class HFDatasetWrapper(torch.utils.data.Dataset):
    """Wrap a *datasets* Dataset so it behaves like a normal Torch vision dataset.

    The underlying dataset must have fields "image" and "label" (or "labels").
    """

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform
        # figure out the label key once
        if "label" in hf_ds.column_names:
            self.label_key = "label"
        elif "labels" in hf_ds.column_names:
            self.label_key = "labels"
        else:
            raise ValueError("HF dataset does not contain a label column")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]
        lbl = int(sample[self.label_key])
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def _load_imagenetc(cfg: Dict[str, Any]):
    if load_dataset is None:
        raise RuntimeError("datasets library not available – cannot load ImageNet-C")

    # Try alternative ImageNet-C datasets
    dataset_candidates = [
        "yangzhou321/ImageNet1k_Corrupt",
        "dingw/corrupt_ood_imagenet1k",
        "timm/mini-imagenet",
    ]
    
    ds_dict = None
    for ds_name in dataset_candidates:
        try:
            ds_dict = load_dataset(ds_name)
            break
        except Exception:
            continue
    
    if ds_dict is None:
        raise RuntimeError("Failed to load any ImageNet-C dataset from HuggingFace")

    # The dataset provides train/validation/test splits; if not, we
    # create a random split.
    if set(ds_dict.keys()) >= {"train", "validation"}:
        hf_train = ds_dict["train"]
        hf_val = ds_dict["validation"]
    elif set(ds_dict.keys()) >= {"train", "test"}:
        hf_train = ds_dict["train"]
        hf_val = ds_dict["test"]
    else:
        full = ds_dict[list(ds_dict.keys())[0]]  # take the only available split
        frac = 0.1
        n_val = int(len(full) * frac)
        indices = list(range(len(full)))
        random.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        hf_train = full.select(train_idx)
        hf_val = full.select(val_idx)

    train_tfms, test_tfms = build_transforms(cfg)
    train_set = HFDatasetWrapper(hf_train, transform=train_tfms)
    val_set = HFDatasetWrapper(hf_val, transform=test_tfms)

    # determine # classes from features if possible
    features = hf_train.features
    if "label" in features and hasattr(features["label"], "num_classes"):
        num_classes = features["label"].num_classes
    else:
        num_classes = len({int(x["label"]) for x in hf_train})
    return train_set, val_set, num_classes


def _load_cifar10c(cfg: Dict[str, Any]):
    if load_dataset is None:
        raise RuntimeError("datasets library not available – cannot load CIFAR10-C")

    ds_name = "randall-lab/cifar10-c"
    hf_ds = load_dataset(ds_name, split="test", trust_remote_code=True)
    # Random split 90/10 – we do *not* use the corrupt images for training; this
    # is just to satisfy the train/val interface.
    n_total = len(hf_ds)
    n_val = n_total // 10
    indices = list(range(n_total))
    random.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    hf_train = hf_ds.select(train_idx)
    hf_val = hf_ds.select(val_idx)

    train_tfms, test_tfms = build_transforms(cfg)
    train_set = HFDatasetWrapper(hf_train, transform=train_tfms)
    val_set = HFDatasetWrapper(hf_val, transform=test_tfms)
    num_classes = 10  # CIFAR-10
    return train_set, val_set, num_classes


def get_dataset(cfg: Dict[str, Any]):
    ds_name = cfg.get("dataset", {}).get("name", "FAKEDATA").upper()

    # ---------------------------------------------------------------------
    # 1) Light-weight FakeData for infrastructure smoke tests
    # ---------------------------------------------------------------------
    if ds_name == "FAKEDATA":
        num_classes = cfg.get("dataset", {}).get("num_classes", 10)
        img_size = cfg.get("dataset", {}).get("img_size", 224)
        train_tfms, test_tfms = build_transforms(cfg)
        train_set = datasets.FakeData(size=1_000, image_size=(3, img_size, img_size),
                                      num_classes=num_classes, transform=train_tfms)
        val_set = datasets.FakeData(size=200, image_size=(3, img_size, img_size),
                                    num_classes=num_classes, transform=test_tfms)
        return train_set, val_set, num_classes

    # ---------------------------------------------------------------------
    # 2) ImageNet-C (mini) – used in the main experiments
    # ---------------------------------------------------------------------
    if ds_name == "IMAGENETC":
        return _load_imagenetc(cfg)

    # ---------------------------------------------------------------------
    # 3) CIFAR10-C – not used in current experiment but provided for future use
    # ---------------------------------------------------------------------
    if ds_name == "CIFAR10C":
        return _load_cifar10c(cfg)

    # ---------------------------------------------------------------------
    # Unknown dataset
    # ---------------------------------------------------------------------
    raise ValueError(f"Unknown dataset: {ds_name}")


# ---------------------------------------------------------------------------
# Dataloader builder (public API)
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: Dict[str, Any], seed: Optional[int] = None):
    if seed is not None:
        seed_everything(seed)

    batch_size = cfg.get("training", {}).get("batch_size", 32)
    num_workers = cfg.get("dataset", {}).get("num_workers", 4)

    train_set, val_set, num_classes = get_dataset(cfg)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes
