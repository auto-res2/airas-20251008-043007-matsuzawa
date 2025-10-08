"""src/evaluate.py
Aggregates results from all run-variation sub-directories under
<results_dir> and creates comparative figures + JSON summary.
It also demonstrates *model loading* by re-evaluating each saved model on
its validation split (sanity check / reproducibility guard).
"""
from __future__ import annotations
import argparse, json, os, pathlib
from typing import Dict, List
import yaml

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .preprocess import get_dataloaders
from .model import get_model

plt.switch_backend("Agg")  # headless environments

# ------------------------------ helpers ------------------------------------ #

def load_results_json(path: os.PathLike) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(d: os.PathLike):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


# ------------------------------ evaluation ----------------------------------#

def create_lineplot(df: pd.DataFrame, results_dir: str):
    img_dir = os.path.join(results_dir, "images")
    ensure_dir(img_dir)

    plt.figure(figsize=(6, 4))
    for run_id, g in df.groupby("run_id"):
        plt.plot(g["epoch"], g["val_acc"], label=run_id)
        # annotate final value
        last_row = g.iloc[-1]
        plt.annotate(f"{last_row['val_acc']:.3f}",
                     (last_row["epoch"], last_row["val_acc"]),
                     textcoords="offset points", xytext=(0, 5), ha="center")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(img_dir, "accuracy_curve.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    return fname


def create_barplot(summary: pd.DataFrame, results_dir: str):
    img_dir = os.path.join(results_dir, "images")
    ensure_dir(img_dir)

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=summary, x="run_id", y="best_val_acc")
    for p, acc in zip(ax.patches, summary["best_val_acc"]):
        height = p.get_height()
        ax.annotate(f"{acc:.3f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9)
    plt.ylabel("Best Validation Accuracy")
    plt.xlabel("Run ID")
    plt.title("Best Validation Accuracy Across Runs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = os.path.join(img_dir, "accuracy_comparison.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return fname


def recompute_confusion(run_dir: str, cfg: Dict, device: torch.device):
    """Load saved *best* model and compute confusion matrix on val set."""
    ckpt_path = os.path.join(run_dir, "model_best.pth")
    if not os.path.exists(ckpt_path):
        return None  # skip if missing (should not happen)

    # Rebuild dataloader to guarantee same split
    _, val_loader, num_classes = get_dataloaders(cfg, seed=42)  # deterministic
    model = get_model(cfg, num_classes=num_classes)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds)
    return cm


def plot_confusion(cm, run_id: str, results_dir: str):
    img_dir = os.path.join(results_dir, "images")
    ensure_dir(img_dir)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix – {run_id}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fname = os.path.join(img_dir, f"confusion_{run_id}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return fname


# --------------------------------------------------------------------------- #

def run(results_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs = [d for d in pathlib.Path(results_dir).iterdir() if d.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No run sub-directories found under {results_dir}")

    # -------------------------- aggregate metrics --------------------------- #
    per_epoch_records: List[Dict] = []
    summary_records: List[Dict] = []

    for rd in run_dirs:
        res_path = rd / "results.json"
        if not res_path.exists():
            print(f"Warning: results.json missing for {rd.name} – skipping")
            continue
        res = load_results_json(res_path)
        for em in res["epoch_metrics"]:
            per_epoch_records.append({**em, "run_id": res["run_id"]})
        summary_records.append({
            "run_id": res["run_id"],
            "best_val_acc": res["best_val_acc"],
            "final_val_acc": res["final_val_acc"],
        })

        # ------------------ recompute confusion matrix ---------------------- #
        cfg_path = rd / "run_config.yaml"
        cfg = yaml.safe_load(open(cfg_path))
        cm = recompute_confusion(str(rd), cfg, device)
        if cm is not None:
            plot_confusion(cm, res["run_id"], results_dir)

    df_epoch = pd.DataFrame(per_epoch_records)
    df_summary = pd.DataFrame(summary_records)

    fig1 = create_lineplot(df_epoch, results_dir)
    fig2 = create_barplot(df_summary, results_dir)

    output = {
        "n_runs": len(df_summary),
        "figures": [os.path.basename(fig1), os.path.basename(fig2)] +
                    [f for f in os.listdir(os.path.join(results_dir, "images")) if f.startswith("confusion_")],
        "comparative_metrics": df_summary.to_dict(orient="records"),
    }
    print(json.dumps(output))


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir)
