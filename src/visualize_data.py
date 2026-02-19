import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from PIL import Image

# Surgical phase names from HeiCo (Online-only Table 1)
# https://pmc.ncbi.nlm.nih.gov/articles/PMC8042116/
PHASE_NAMES = {
    0: "Preparation & Orientation",
    1: "Dissection of Lymph Nodes & Vessels",
    2: "Retroperitoneal Prep. (Lower Pancreatic Border)",
    3: "Retroperitoneal Prep. (Duodenum & Pancreatic Head)",
    4: "Mobilization of Sigmoid & Descending Colon",
    5: "Mobilization of Splenic Flexure",
    6: "Mobilization of Transverse Colon",
    7: "Mobilization of Ascending Colon",
    8: "Dissection & Resection of Rectum",
    9: "Extra-abdominal Prep. of Anastomosis",
    10: "Intra-abdominal Prep. of Anastomosis",
    11: "Creation of Stoma",
    12: "Finalization of Operation",
    13: "Exception",
}

PHASE_COLORS = plt.cm.Set3(np.linspace(0, 1, len(PHASE_NAMES)))


def load_phase_data(video_dir: Path):
    """Load per-frame phase annotations."""
    phase_files = list(video_dir.glob("*_phase.csv"))
    if not phase_files:
        print("No phase CSV found.")
        return None
    df = pd.read_csv(phase_files[0], header=None, names=["frame", "phase"])
    return df


def load_device_data(video_dir: Path):
    """Load per-frame device/sensor data."""
    device_files = list(video_dir.glob("*_device.csv"))
    if not device_files:
        print("No device CSV found.")
        return None
    df = pd.read_csv(device_files[0], header=None)
    df.columns = ["frame"] + [f"sensor_{i}" for i in range(df.shape[1] - 1)]
    return df


def get_available_frames(video_dir: Path):
    """Get sorted list of annotated frame directories."""
    seg_dir = video_dir / "Instrument segmentations"
    if not seg_dir.exists():
        return []
    frames = []
    for d in seg_dir.iterdir():
        if d.is_dir():
            try:
                frames.append(int(d.name))
            except ValueError:
                continue
    return sorted(frames)


def plot_phase_timeline(phase_df, ax):
    """Plot surgical phase as a color-coded timeline."""
    frames = phase_df["frame"].values
    phases = phase_df["phase"].values
    unique_phases = sorted(phase_df["phase"].unique())

    cmap = ListedColormap([PHASE_COLORS[p] for p in unique_phases])

    # Draw as a horizontal color bar
    phase_img = phases.reshape(1, -1)
    ax.imshow(
        phase_img,
        aspect="auto",
        cmap=cmap,
        vmin=min(unique_phases) - 0.5,
        vmax=max(unique_phases) + 0.5,
        extent=[frames[0], frames[-1], 0, 1],
    )
    ax.set_yticks([])
    ax.set_xlabel("Frame")
    ax.set_title("Surgical Phase Timeline")

    # Legend
    patches = [
        mpatches.Patch(color=PHASE_COLORS[p], label=f"{p}: {PHASE_NAMES.get(p, f'Phase {p}')}")
        for p in unique_phases
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)


def plot_phase_durations(phase_df, ax):
    """Bar chart of phase durations (in frames)."""
    counts = phase_df.groupby("phase")["frame"].count()
    unique_phases = sorted(counts.index)
    colors = [PHASE_COLORS[p] for p in unique_phases]
    labels = [f"{p}: {PHASE_NAMES.get(p, '?')}" for p in unique_phases]

    ax.barh(labels, [counts[p] for p in unique_phases], color=colors)
    ax.set_xlabel("Number of Frames")
    ax.set_title("Phase Durations")


def plot_device_data(device_df, phase_df, ax):
    """Plot a few interesting sensor channels over time with phase background."""
    frames = device_df["frame"].values
    # Pick a few sensor columns that vary
    sensor_cols = [c for c in device_df.columns if c.startswith("sensor_")]
    stds = device_df[sensor_cols].std()
    top_sensors = stds.nlargest(4).index.tolist()

    for col in top_sensors:
        vals = device_df[col].values
        ax.plot(frames[::100], vals[::100], label=col, alpha=0.8, linewidth=0.8)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Sensor Value")
    ax.set_title("Device Sensor Data (top 4 by variance)")
    ax.legend(fontsize=7)


def plot_sample_frames(video_dir, phase_df, available_frames, axes):
    """Show one sample frame per phase."""
    seg_dir = video_dir / "Instrument segmentations"
    unique_phases = sorted(phase_df["phase"].unique())
    available_set = set(available_frames)

    # For each phase find closest available frame
    samples = {}
    for phase in unique_phases:
        phase_frames = phase_df[phase_df["phase"] == phase]["frame"].values
        mid = phase_frames[len(phase_frames) // 2]
        # Find closest available frame
        closest = min(available_frames, key=lambda f: abs(f - mid)) if available_frames else None
        if closest is not None:
            samples[phase] = closest

    for idx, ax in enumerate(axes.flat):
        if idx >= len(unique_phases):
            ax.axis("off")
            continue
        phase = unique_phases[idx]
        frame_num = samples.get(phase)
        if frame_num is None:
            ax.axis("off")
            continue
        img_path = seg_dir / str(frame_num) / "raw.png"
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
        ax.set_title(f"Phase {phase}: {PHASE_NAMES.get(phase, '?')}\n(frame {frame_num})", fontsize=7)
        ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Visualize HeiCo surgery data")
    parser.add_argument("--video_dir", type=str, help="Path to a video directory (e.g. data/Proctocolectomy/1)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        sys.exit(1)

    # Load data
    phase_df = load_phase_data(video_dir)
    device_df = load_device_data(video_dir)
    available_frames = get_available_frames(video_dir)

    print(f"Video directory: {video_dir}")
    if phase_df is not None:
        print(f"Phase data: {len(phase_df)} frames, phases: {sorted(phase_df['phase'].unique())}")
    if device_df is not None:
        print(f"Device data: {len(device_df)} frames, {device_df.shape[1] - 1} sensor columns")
    print(f"Annotated keyframes: {len(available_frames)}")

    # --- Figure 1: Timeline + durations + sensor data ---
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 2, 2]})
    fig1.suptitle(f"HeiCo Data Overview — {video_dir.name}", fontsize=13)

    if phase_df is not None:
        plot_phase_timeline(phase_df, axes1[0])
        plot_phase_durations(phase_df, axes1[1])
    if device_df is not None:
        plot_device_data(device_df, phase_df, axes1[2])

    fig1.tight_layout()

    # --- Figure 2: Sample frames per phase ---
    if phase_df is not None and available_frames:
        n_phases = len(phase_df["phase"].unique())
        ncols = min(4, n_phases)
        nrows = (n_phases + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        fig2.suptitle("Sample Frames per Surgical Phase", fontsize=13)
        if nrows == 1 and ncols == 1:
            axes2 = np.array([[axes2]])
        elif nrows == 1 or ncols == 1:
            axes2 = axes2.reshape(nrows, ncols)
        plot_sample_frames(video_dir, phase_df, available_frames, axes2)
        fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
