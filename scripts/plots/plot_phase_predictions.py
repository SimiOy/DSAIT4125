"""
Plot model predictions vs ground truth as a phase timeline (colored bars).

Usage:
    python scripts/plots/plot_phase_predictions.py \
        results/ImageNet_Surgformer/Cholec80/testing/0.txt \
        results/EndoVIT_Surgformer/Cholec80/testing/0.txt \
        --output results/comparison/phase_timeline_cholec80.png \
        --title "Surgformer — Cholec80 test set phase timeline"

Each row is a colored stripe where the color encodes the surgical phase.
Three rows: Ground Truth | ImageNet Surgformer | EndoVIT Surgformer.
X-axis = frames (no tick labels — too dense).
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Dataset phase metadata
# ---------------------------------------------------------------------------
DATASETS = {
    "cholec80": {
        "phase_names": [
            "Preparation",
            "CalotTriangle\nDissection",
            "Clipping &\nCutting",
            "Gallbladder\nDissection",
            "Gallbladder\nPackaging",
            "Cleaning /\nCoagulation",
            "Gallbladder\nRetraction",
        ],
        "colors": [
            "#b3cde0",  # 0 Preparation        — pastel blue
            "#fbb4ae",  # 1 CalotTriangle       — pastel orange
            "#fed9a6",  # 2 Clipping            — pastel red/pink
            "#b5ead7",  # 3 GB Dissection       — pastel mint
            "#e5d8bd",  # 4 GB Packaging        — pastel green
            "#fddaec",  # 5 Cleaning            — pastel yellow
            "#decbe4",  # 6 GB Retraction       — pastel lavender
        ],
    },
    "m2cai16": {
        "phase_names": [
            "Trocar\nPlacement",
            "Preparation",
            "CalotTriangle\nDissection",
            "Clipping &\nCutting",
            "Gallbladder\nDissection",
            "Gallbladder\nPackaging",
            "Cleaning /\nCoagulation",
            "Gallbladder\nRetraction",
        ],
        "colors": [
            "#b3cde0",  # 0 Trocar Placement    — pastel steel blue
            "#ccebc5",  # 1 Preparation         — pastel sage green
            "#fbb4ae",  # 2 CalotTriangle       — pastel salmon
            "#fed9a6",  # 3 Clipping            — pastel peach
            "#ffffcc",  # 4 GB Dissection       — pastel lemon
            "#e5d8bd",  # 5 GB Packaging        — pastel tan
            "#fddaec",  # 6 Cleaning            — pastel rose
            "#decbe4",  # 7 GB Retraction       — pastel lilac
        ],
    },
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_predictions(path: Path):
    """Return dict {global_idx: (pred, gt)} parsed from a test output file."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lb = line.find("[")
            rb = line.find("]")
            if lb == -1 or rb == -1:
                continue
            prefix = line[:lb].split()
            if len(prefix) < 1:
                continue
            try:
                global_idx = int(prefix[0])
            except ValueError:
                continue  # skip header / accuracy line
            logits_str = line[lb:rb + 1]
            gt_str = line[rb + 1:].strip()
            try:
                gt = int(gt_str)
                logits = ast.literal_eval(logits_str)
            except Exception:
                continue
            pred = int(np.argmax(logits))
            data[global_idx] = (pred, gt)
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_timeline(
    gt: np.ndarray,
    pred_imagenet: np.ndarray,
    pred_endovit: np.ndarray,
    title: str,
    output: Path,
    phase_names: list,
    phase_colors: list,
):
    n = len(gt)
    num_phases = len(phase_names)
    cmap = ListedColormap(phase_colors)

    # Build a 3 x N image: row 0 = GT, row 1 = ImageNet, row 2 = EndoVIT
    img = np.stack([gt, pred_imagenet, pred_endovit]).astype(float)  # (3, N)

    row_labels = ["Ground\nTruth", "ImageNet\nSurgformer", "EndoViT\nSurgformer"]

    fig, ax = plt.subplots(figsize=(18, 3.2))
    fig.subplots_adjust(left=0.12, right=0.88, top=0.82, bottom=0.08)

    ax.imshow(
        img,
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=num_phases - 0.5,
        interpolation="nearest",
    )

    # Y-axis: row labels
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.tick_params(axis="y", length=0)

    # X-axis: no tick labels (too dense), just a frame-count label
    ax.set_xticks([])
    ax.set_xlabel(f"Frame index  (0 → {n - 1})", fontsize=8, labelpad=4)

    # Horizontal dividers between rows
    for y in [0.5, 1.5]:
        ax.axhline(y, color="white", linewidth=1.5)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Legend
    patches = [
        mpatches.Patch(color=phase_colors[i], label=f"Phase {i}: {phase_names[i].replace(chr(10), ' ')}")
        for i in range(num_phases)
    ]
    ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7.5,
        framealpha=0.9,
        title="Surgical Phase",
        title_fontsize=8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase timeline: predictions vs ground truth")
    parser.add_argument("imagenet_file", type=Path, help="ImageNet Surgformer test output (0.txt)")
    parser.add_argument("endovit_file",  type=Path, help="EndoViT  Surgformer test output (0.txt)")
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/comparison/phase_timeline_cholec80.png"),
    )
    parser.add_argument(
        "--title", type=str,
        default="Surgformer — Cholec80 test set  |  phase predictions vs ground truth",
    )
    parser.add_argument(
        "--dataset", type=str, choices=list(DATASETS.keys()), default="cholec80",
        help="Dataset name (determines phase labels and colours)",
    )
    args = parser.parse_args()

    print("Parsing ImageNet predictions …")
    data_in = parse_predictions(args.imagenet_file)

    print("Parsing EndoViT predictions …")
    data_ev = parse_predictions(args.endovit_file)

    # Align on the shared frame indices that appear in BOTH files
    shared_indices = sorted(set(data_in.keys()) & set(data_ev.keys()))
    if not shared_indices:
        raise ValueError("No shared frame indices between the two files.")

    gt      = np.array([data_in[i][1] for i in shared_indices], dtype=np.int8)
    pred_in = np.array([data_in[i][0] for i in shared_indices], dtype=np.int8)
    pred_ev = np.array([data_ev[i][0] for i in shared_indices], dtype=np.int8)
    n = len(shared_indices)

    print(f"Shared frames: {n}  (ImageNet total: {len(data_in)}, EndoViT total: {len(data_ev)})")
    print(f"ImageNet  accuracy: {(pred_in == gt).mean() * 100:.2f}%")
    print(f"EndoViT   accuracy: {(pred_ev == gt).mean() * 100:.2f}%")

    ds = DATASETS[args.dataset]
    plot_timeline(gt, pred_in, pred_ev, args.title, args.output, ds["phase_names"], ds["colors"])


if __name__ == "__main__":
    main()
