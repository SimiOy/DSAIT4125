"""
Compare ImageNet-Surgformer vs EndoViT-Surgformer training curves on the same plot.

Manual usage (one dataset at a time):
    python scripts/plots/plot_training_curves.py \\
        --imagenet-logs results/ImageNet_Surgformer/Cholec80/training/*.out \\
        --endovit-logs  results/EndoVIT_Surgformer/Cholec80/training/*.out \\
        --output        results/comparison/cholec80_training_curves.png \\
        --title         "Cholec80 — ImageNet vs EndoViT pretrain"

Auto mode (generates all 4 dataset comparison plots at once):
    python scripts/plots/plot_training_curves.py --auto

Multiple .out files per model are merged in order; duplicate epochs keep the first occurrence.
"""

import re
import argparse
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

RE_EPOCH_END   = re.compile(r"^Epoch: \[(\d+)\] Total time:")
RE_TRAIN_STATS = re.compile(r"^Averaged stats:.*?\bloss:\s*[\d.]+\s*\(([\d.]+)\)")
RE_VAL_STATS   = re.compile(r"^\* Acc@1\s+([\d.]+)\s+Acc@5\s+([\d.]+)\s+loss\s+([\d.]+)")


def parse_log(path: Path) -> dict[int, dict]:
    """Return {epoch: {train_loss, val_acc1, val_acc5, val_loss}}."""
    records: OrderedDict[int, dict] = OrderedDict()
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
                val_acc1  = float(m.group(1))
                val_acc5  = float(m.group(2))
                val_loss  = float(m.group(3))
                if current_epoch not in records:
                    records[current_epoch] = {
                        "train_loss": pending_train_loss,
                        "val_acc1":   val_acc1,
                        "val_acc5":   val_acc5,
                        "val_loss":   val_loss,
                    }
                pending_train_loss = None
                current_epoch = None

    return records


def merge_records(*record_dicts) -> OrderedDict:
    """Merge multiple parsed dicts; first occurrence of each epoch wins."""
    merged: OrderedDict[int, dict] = OrderedDict()
    for rd in record_dicts:
        for epoch, data in rd.items():
            if epoch not in merged:
                merged[epoch] = data
    return OrderedDict(sorted(merged.items()))


def load_model_records(log_paths: list[Path]) -> OrderedDict:
    return merge_records(*[parse_log(p) for p in log_paths])


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Each metric gets a (dark, light) pair.
# EndoViT  → dark   (solid lines)
# ImageNet → light  (dashed lines, same hue)

COLORS = {
    "train_loss": ("#1f77b4", "#aec7e8"),   # blue family
    "val_loss":   ("#d62728", "#f7a6a6"),   # red family
    "val_acc1":   ("#2ca02c", "#98df8a"),   # green family
    "val_acc5":   ("#9467bd", "#c5b0d5"),   # purple family
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_model(ax_loss, ax_acc, records: OrderedDict, label_prefix: str,
                dark: bool, show_train_loss: bool = True):
    """Plot one model's curves onto the supplied axes."""
    if not records:
        return

    style   = "-"  if dark else "--"
    alpha   = 1.0  if dark else 0.85
    ms      = 3    if dark else 2.5

    c_tl  = COLORS["train_loss"][0 if dark else 1]
    c_vl  = COLORS["val_loss"][0   if dark else 1]
    c_a1  = COLORS["val_acc1"][0   if dark else 1]
    c_a5  = COLORS["val_acc5"][0   if dark else 1]

    epochs     = sorted(records.keys())
    train_loss = [records[e]["train_loss"] for e in epochs]
    val_loss   = [records[e]["val_loss"]   for e in epochs]
    val_acc1   = [records[e]["val_acc1"]   for e in epochs]
    val_acc5   = [records[e]["val_acc5"]   for e in epochs]

    best_epoch = max(epochs, key=lambda e: records[e]["val_acc1"])
    best_acc1  = records[best_epoch]["val_acc1"]

    # Loss panel
    if show_train_loss and any(v is not None for v in train_loss):
        ax_loss.plot(epochs, train_loss,
                     label=f"{label_prefix} Train loss",
                     color=c_tl, linestyle=style, marker="o",
                     markersize=ms, linewidth=1.5, alpha=alpha)
    ax_loss.plot(epochs, val_loss,
                 label=f"{label_prefix} Val loss",
                 color=c_vl, linestyle=style, marker="s",
                 markersize=ms, linewidth=1.5, alpha=alpha)

    # Accuracy panel
    ax_acc.plot(epochs, val_acc1,
                label=f"{label_prefix} Val Acc@1",
                color=c_a1, linestyle=style, marker="o",
                markersize=ms, linewidth=1.5, alpha=alpha)
    ax_acc.plot(epochs, val_acc5,
                label=f"{label_prefix} Val Acc@5",
                color=c_a5, linestyle=style, marker="s",
                markersize=ms, linewidth=1.5, alpha=alpha)
    ax_acc.axvline(best_epoch,
                   color=c_a1, linestyle=":",
                   linewidth=1.2 if dark else 1.0,
                   label=f"{label_prefix} Best Acc@1={best_acc1:.2f}% (ep {best_epoch})")


def plot_comparison(imagenet_records: OrderedDict,
                    endovit_records:  OrderedDict,
                    title:  str,
                    output: Path):
    """Generate a 1×2 comparison figure with both models overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax_loss, ax_acc = axes

    # EndoViT first (dark colours, solid)
    _plot_model(ax_loss, ax_acc, endovit_records,  label_prefix="EndoViT",  dark=True)
    # ImageNet second (light colours, dashed)
    _plot_model(ax_loss, ax_acc, imagenet_records, label_prefix="ImageNet", dark=False)

    # Loss panel formatting
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_loss.grid(True, alpha=0.3)

    # Accuracy panel formatting
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Validation Accuracy")
    ax_acc.legend(fontsize=8)
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Auto mode — hard-coded dataset configs
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent / "results"

AUTO_CONFIGS = [
    {
        "title":  "Cholec80 — ImageNet vs EndoViT pretrain",
        "output": ROOT / "comparison" / "cholec80_training_curves.png",
        "imagenet_logs": sorted((ROOT / "ImageNet_Surgformer" / "Cholec80" / "training").glob("*.out")),
        "endovit_logs":  sorted((ROOT / "EndoVIT_Surgformer"  / "Cholec80" / "training").glob("*.out")),
    },
    {
        "title":  "M2CAI16 full — ImageNet vs EndoViT pretrain",
        "output": ROOT / "comparison" / "m2cai16_full_training_curves.png",
        "imagenet_logs": sorted((ROOT / "ImageNet_Surgformer" / "M2CAI16" / "full"    / "training").glob("*.out")),
        "endovit_logs":  sorted((ROOT / "EndoVIT_Surgformer"  / "M2CAI16" / "full"    / "training").glob("*.out")),
    },
    {
        "title":  "M2CAI16 half — ImageNet vs EndoViT pretrain",
        "output": ROOT / "comparison" / "m2cai16_half_training_curves.png",
        "imagenet_logs": sorted((ROOT / "ImageNet_Surgformer" / "M2CAI16" / "half"    / "training").glob("*.out")),
        "endovit_logs":  sorted((ROOT / "EndoVIT_Surgformer"  / "M2CAI16" / "half"    / "training").glob("*.out")),
    },
    {
        "title":  "M2CAI16 quarter — ImageNet vs EndoViT pretrain",
        "output": ROOT / "comparison" / "m2cai16_quarter_training_curves.png",
        "imagenet_logs": sorted((ROOT / "ImageNet_Surgformer" / "M2CAI16" / "quarter" / "training").glob("*.out")),
        "endovit_logs":  sorted((ROOT / "EndoVIT_Surgformer"  / "M2CAI16" / "quarter" / "training").glob("*.out")),
    },
]


def run_auto():
    for cfg in AUTO_CONFIGS:
        inet = cfg["imagenet_logs"]
        evit = cfg["endovit_logs"]

        inet_records = load_model_records(inet) if inet else OrderedDict()
        evit_records = load_model_records(evit) if evit else OrderedDict()

        if not inet_records and not evit_records:
            print(f"[SKIP] No logs found for: {cfg['title']}")
            continue

        _summarise("ImageNet", inet_records)
        _summarise("EndoViT",  evit_records)

        plot_comparison(inet_records, evit_records, cfg["title"], cfg["output"])


def _summarise(label: str, records: OrderedDict):
    if not records:
        print(f"  {label}: no data")
        return
    best = max(records, key=lambda e: records[e]["val_acc1"])
    print(f"  {label}: {len(records)} epochs, "
          f"best Val Acc@1 = {records[best]['val_acc1']:.2f}% @ ep {best}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot Surgformer training curves — ImageNet vs EndoViT comparison"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-generate all 3 dataset comparison plots using standard directory layout"
    )
    parser.add_argument(
        "--imagenet-logs", nargs="*", type=Path, default=[],
        metavar="FILE",
        help=".out log files for the ImageNet-pretrained model"
    )
    parser.add_argument(
        "--endovit-logs", nargs="*", type=Path, default=[],
        metavar="FILE",
        help=".out log files for the EndoViT-pretrained model"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/comparison/training_curves.png"),
        help="Output PNG path (manual mode)"
    )
    parser.add_argument(
        "--title", type=str,
        default="Surgformer HTA-KCA — ImageNet vs EndoViT",
        help="Plot title (manual mode)"
    )
    args = parser.parse_args()

    if args.auto:
        run_auto()
        return

    if not args.imagenet_logs and not args.endovit_logs:
        parser.error("Provide --imagenet-logs and/or --endovit-logs, or use --auto")

    inet_records = load_model_records(args.imagenet_logs) if args.imagenet_logs else OrderedDict()
    evit_records = load_model_records(args.endovit_logs)  if args.endovit_logs  else OrderedDict()

    _summarise("ImageNet", inet_records)
    _summarise("EndoViT",  evit_records)

    plot_comparison(inet_records, evit_records, args.title, args.output)


if __name__ == "__main__":
    main()
