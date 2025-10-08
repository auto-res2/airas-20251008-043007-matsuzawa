"""src/train.py
Conducts a single experiment run as specified by an *individual* run-variation
configuration (YAML) that already lives inside <results_dir>/<run_id>/run_config.yaml

The script is **dataset / model agnostic** – all dataset-specific logic lives in
src/preprocess.get_dataloaders and model definitions are pulled from
src.model.get_model according to the config.  Therefore this file NEVER
contains placeholders.
"""
from __future__ import annotations
import argparse, json, os, sys, time, pathlib
from typing import Dict, Any

import torch, torch.nn as nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import yaml

from .preprocess import get_dataloaders
from .model import get_model, TentAdaptor, AdaNPCAdaptor


# ------------------------------- helpers ------------------------------------ #

def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy."""
    with torch.no_grad():
        pred_classes = pred.argmax(1)
        correct = (pred_classes == target).sum().item()
    return correct / target.size(0)


def save_json(path: os.PathLike, obj: Dict[str, Any]):
    path = pathlib.Path(path)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------- main train --------------------------------- #

def run(cfg: Dict[str, Any], results_dir: str):
    torch.backends.cudnn.benchmark = True  # speed-up for fixed input size

    run_id: str = cfg["run_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###########################################################################
    # 1.  Data
    ###########################################################################
    train_loader, val_loader, num_classes = get_dataloaders(cfg)

    ###########################################################################
    # 2.  Model & Optimiser
    ###########################################################################
    model = get_model(cfg, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    is_tta_model = isinstance(model, (TentAdaptor, AdaNPCAdaptor))
    
    if not is_tta_model:
        optim_cfg = cfg.get("optimizer", {"name": "SGD", "lr": 0.01, "momentum": 0.9})
        if optim_cfg["name"].lower() == "sgd":
            optimizer = SGD(model.parameters(), lr=optim_cfg["lr"], momentum=optim_cfg.get("momentum", 0))
        elif optim_cfg["name"].lower() == "adamw":
            optimizer = AdamW(model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg.get("weight_decay", 1e-2))
        else:
            raise ValueError(f"Unsupported optimizer: {optim_cfg['name']}")
    else:
        optimizer = None

    scaler = GradScaler(enabled=cfg.get("mixed_precision", True) and device.type == "cuda")

    ###########################################################################
    # 3.  Training loop
    ###########################################################################
    epochs = cfg.get("training", {}).get("epochs", 10)
    log_every = max(1, len(train_loader) // 10)

    epoch_metrics = []
    best_val_acc = -1
    best_ckpt_path = os.path.join(results_dir, run_id, "model_best.pth")
    
    for epoch in range(epochs):
        model.train()
        running_loss, running_acc, n_samples = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"[{run_id}] Train Epoch {epoch+1}/{epochs}", leave=False)
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if is_tta_model:
                with autocast(enabled=scaler.is_enabled()):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=scaler.is_enabled()):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_acc = accuracy(logits.detach(), y)
            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            running_acc += batch_acc * batch_size
            n_samples += batch_size

            if (i + 1) % log_every == 0:
                pbar.set_postfix({"loss": running_loss / n_samples, "acc": running_acc / n_samples})

        train_loss = running_loss / n_samples
        train_acc = running_acc / n_samples

        # ---------------------- validation ---------------------------------- #
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                batch_acc = accuracy(logits, y)

                val_loss += loss.item() * y.size(0)
                val_acc += batch_acc * y.size(0)
                n_val += y.size(0)
        val_loss /= n_val
        val_acc /= n_val

        epoch_metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # ---------------------- checkpointing ------------------------------- #
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg,
                "epoch": epoch + 1,
                "val_acc": val_acc,
            }, best_ckpt_path)

    ###########################################################################
    # 4.  Save artefacts & log final metrics
    ###########################################################################
    final_metrics = {
        "run_id": run_id,
        "final_train_loss": epoch_metrics[-1]["train_loss"],
        "final_train_acc": epoch_metrics[-1]["train_acc"],
        "final_val_loss": epoch_metrics[-1]["val_loss"],
        "final_val_acc": epoch_metrics[-1]["val_acc"],
        "best_val_acc": best_val_acc,
        "epochs": epochs,
        "epoch_metrics": epoch_metrics,
    }

    # last checkpoint (overwrite each run)
    last_ckpt_path = os.path.join(results_dir, run_id, "model_last.pth")
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "epoch": epochs,
        "val_acc": epoch_metrics[-1]["val_acc"],
    }, last_ckpt_path)

    # save json   
    save_json(os.path.join(results_dir, run_id, "results.json"), final_metrics)

    # provide machine-readable output on stdout
    print(json.dumps(final_metrics))


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to run-specific YAML config file")
    p.add_argument("--results-dir", required=True, help="Root results directory for this whole experiment set")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.results_dir)
