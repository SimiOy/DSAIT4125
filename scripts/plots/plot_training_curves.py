"""
Usage:
    python scripts/plot_training_curves.py \
        results/ImageNet_Surgformer/Cholec80/training/surgformer-train.9310069.out \
        results/ImageNet_Surgformer/Cholec80/training/surgformer-train.9316723.out \
        --output results/ImageNet_Surgformer/Cholec80/training/training_curves.png \
        --title "Surgformer HTA-KCA — ImageNet pretrain — Cholec80"

Multiple .out files are merged in order; duplicate epochs keep the first occurrence.
"""

import re
import argparse
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Epoch: [N] Total time: ...
RE_EPOCH_END = re.compile(r"^Epoch: \[(\d+)\] Total time:")

# Averaged stats: lr: X  min_lr: X  loss: X (Y)  ...
RE_TRAIN_STATS = re.compile(
    r"^Averaged stats:.*?\bloss:\s*[\d.]+\s*\(([\d.]+)\)"
)

# * Acc@1 XX.XXX Acc@5 YY.YYY loss Z.ZZZ
RE_VAL_STATS = re.compile(
    r"^\* Acc@1\s+([\d.]+)\s+Acc@5\s+([\d.]+)\s+loss\s+([\d.]+)"
)


def parse_log(path: Path) -> dict[int, dict]:
    """Return {epoch: {train_loss, val_acc1, val_acc5, val_loss}}."""
    records = OrderedDict()
    current_epoch = None
    pending_train_loss = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()

            m = RE_EPOCH_END.match(line)
            if m:
                current_epoch = int(m.group(1))
                pending_train_loss = None
                continue

            m = RE_TRAIN_STATS.match(line)
            if m and current_epoch is not None:
                pending_train_loss = float(m.group(1))
                continue

            m = RE_VAL_STATS.match(line)
            if m and current_epoch is not None:
                val_acc1 = float(m.group(1))
                val_acc5 = float(m.group(2))
                val_loss = float(m.group(3))
                if current_epoch not in records:
                    records[current_epoch] = {
                        "train_loss": pending_train_loss,
                        "val_acc1": val_acc1,
                        "val_acc5": val_acc5,
                        "val_loss": val_loss,
                    }
                pending_train_loss = None
                current_epoch = None  # reset until next epoch end marker

    return records


def merge_records(*record_dicts) -> OrderedDict:
    """Merge multiple parsed dicts; first occurrence of each epoch wins."""
    merged = OrderedDict()
    for rd in record_dicts:
        for epoch, data in rd.items():
            if epoch not in merged:
                merged[epoch] = data
    return OrderedDict(sorted(merged.items()))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(records: OrderedDict, title: str, output: Path):
    epochs = sorted(records.keys())
    train_loss = [records[e]["train_loss"] for e in epochs]
    val_loss   = [records[e]["val_loss"]   for e in epochs]
    val_acc1   = [records[e]["val_acc1"]   for e in epochs]
    val_acc5   = [records[e]["val_acc5"]   for e in epochs]

    best_epoch = max(epochs, key=lambda e: records[e]["val_acc1"])
    best_acc1  = records[best_epoch]["val_acc1"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- Loss subplot ---
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train loss", marker="o", markersize=3, linewidth=1.5)
    ax.plot(epochs, val_loss,   label="Val loss",   marker="s", markersize=3, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # --- Accuracy subplot ---
    ax = axes[1]
    ax.plot(epochs, val_acc1, label="Val Acc@1", marker="o", markersize=3, linewidth=1.5, color="tab:green")
    ax.plot(epochs, val_acc5, label="Val Acc@5", marker="s", markersize=3, linewidth=1.5, color="tab:olive", linestyle="--")
    ax.axvline(best_epoch, color="tab:red", linestyle=":", linewidth=1.2, label=f"Best Acc@1 = {best_acc1:.2f}% (ep {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {output}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Surgformer training curves from SLURM .out files")
    parser.add_argument("log_files", nargs="+", type=Path, help=".out log files (merged in order)")
    parser.add_argument("--output", type=Path,
                        default=Path("results/ImageNet_Surgformer/Cholec80/training/training_curves.png"),
                        help="Output PNG path")
    parser.add_argument("--title", type=str,
                        default="Surgformer HTA-KCA — ImageNet pretrain — Cholec80",
                        help="Plot title")
    args = parser.parse_args()

    all_records = [parse_log(p) for p in args.log_files]
    merged = merge_records(*all_records)

    if not merged:
        print("No epoch records found. Check log file format.")
        return

    print(f"Parsed {len(merged)} epochs: {min(merged)} → {max(merged)}")
    best = max(merged, key=lambda e: merged[e]["val_acc1"])
    print(f"Best Val Acc@1: {merged[best]['val_acc1']:.2f}% at epoch {best}")

    plot_curves(merged, args.title, args.output)


if __name__ == "__main__":
    main()
