import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from plot_training_curves import load_model_records, plot_comparison, _summarise
from plot_phase_predictions import parse_predictions, plot_timeline, DATASETS

from collections import OrderedDict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "results"
OUT  = ROOT / "comparison"

CURVE_CONFIGS = [
    {
        "title":   "Cholec80 — ImageNet vs EndoViT pretrain",
        "output":  OUT / "cholec80" / "training_curves.png",
        "inet":    ROOT / "ImageNet_Surgformer" / "Cholec80"        / "training",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "Cholec80"        / "training",
    },
    {
        "title":   "M2CAI16 full — ImageNet vs EndoViT pretrain",
        "output":  OUT / "m2cai16" / "full" / "training_curves.png",
        "inet":    ROOT / "ImageNet_Surgformer" / "M2CAI16" / "full"    / "training",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "M2CAI16" / "full"    / "training",
    },
    {
        "title":   "M2CAI16 half — ImageNet vs EndoViT pretrain",
        "output":  OUT / "m2cai16" / "half" / "training_curves.png",
        "inet":    ROOT / "ImageNet_Surgformer" / "M2CAI16" / "half"    / "training",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "M2CAI16" / "half"    / "training",
    },
    {
        "title":   "M2CAI16 quarter — ImageNet vs EndoViT pretrain",
        "output":  OUT / "m2cai16" / "quarter" / "training_curves.png",
        "inet":    ROOT / "ImageNet_Surgformer" / "M2CAI16" / "quarter" / "training",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "M2CAI16" / "quarter" / "training",
    },
    {
        "title":   "MultiBypass140 full — ImageNet vs EndoViT pretrain",
        "output":  OUT / "multibypass140" / "full" / "training_curves.png",
        "inet":    ROOT / "ImageNet_Surgformer" / "multibypass140" / "full"    / "training",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "multibypass140" / "full"    / "training",
    },
    {
        "title":   "MultiBypass140 half — ImageNet vs EndoViT pretrain",
        "output":  OUT / "multibypass140" / "half" / "training_curves.png",
        "inet":    ROOT / "ImageNet_Surgformer" / "multibypass140" / "half"    / "training",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "multibypass140" / "half"    / "training",
    },
    {
        "title":   "MultiBypass140 quarter — ImageNet vs EndoViT pretrain",
        "output":  OUT / "multibypass140" / "quarter" / "training_curves.png",
        "inet":    ROOT / "ImageNet_Surgformer" / "multibypass140" / "quarter" / "training",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "multibypass140" / "quarter" / "training",
    },
]

TIMELINE_CONFIGS = [
    {
        "title":   "Surgformer — Cholec80 test set  |  phase predictions vs ground truth",
        "output":  OUT / "cholec80" / "phase_timeline.png",
        "dataset": "cholec80",
        "inet":    ROOT / "ImageNet_Surgformer" / "Cholec80"        / "testing" / "0.txt",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "Cholec80"        / "testing" / "0.txt",
    },
    {
        "title":   "Surgformer — M2CAI16 full fine-tuning  |  phase predictions vs ground truth",
        "output":  OUT / "m2cai16" / "full" / "phase_timeline.png",
        "dataset": "m2cai16",
        "inet":    ROOT / "ImageNet_Surgformer" / "M2CAI16" / "full" / "testing" / "0.txt",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "M2CAI16" / "full" / "testing" / "0.txt",
    },
    {
        "title":   "Surgformer — MultiBypass140 full fine-tuning  |  phase predictions vs ground truth",
        "output":  OUT / "multibypass140" / "full" / "phase_timeline.png",
        "dataset": "multibypass140",
        "inet":    ROOT / "ImageNet_Surgformer" / "multibypass140" / "full" / "testing" / "0.txt",
        "evit":    ROOT / "EndoVIT_Surgformer"  / "multibypass140" / "full" / "testing" / "0.txt",
    },
]


def run_curves():
    print("\n=== Training curves ===")
    for cfg in CURVE_CONFIGS:
        inet_logs = sorted(cfg["inet"].glob("*.out")) if cfg["inet"].exists() else []
        evit_logs = sorted(cfg["evit"].glob("*.out")) if cfg["evit"].exists() else []
        inet_rec  = load_model_records(inet_logs) if inet_logs else OrderedDict()
        evit_rec  = load_model_records(evit_logs) if evit_logs else OrderedDict()
        if not inet_rec and not evit_rec:
            print(f"  [SKIP] no logs for: {cfg['title']}")
            continue
        print(f"\n{cfg['title']}")
        _summarise("ImageNet", inet_rec)
        _summarise("EndoViT",  evit_rec)
        plot_comparison(inet_rec, evit_rec, cfg["title"], cfg["output"])


def run_timelines():
    print("\n=== Phase timelines ===")
    for cfg in TIMELINE_CONFIGS:
        if not cfg["inet"].exists() or not cfg["evit"].exists():
            print(f"  [SKIP] missing 0.txt for: {cfg['title']}")
            continue
        print(f"\n{cfg['title']}")
        data_in = parse_predictions(cfg["inet"])
        data_ev = parse_predictions(cfg["evit"])
        shared  = sorted(set(data_in) & set(data_ev))
        if not shared:
            print("  [SKIP] no shared frame indices")
            continue
        gt      = np.array([data_in[i][1] for i in shared], dtype=np.int8)
        pred_in = np.array([data_in[i][0] for i in shared], dtype=np.int8)
        pred_ev = np.array([data_ev[i][0] for i in shared], dtype=np.int8)
        print(f"  Frames: {len(shared)} | ImageNet {(pred_in==gt).mean()*100:.2f}%"
              f" | EndoViT {(pred_ev==gt).mean()*100:.2f}%")
        ds = DATASETS[cfg["dataset"]]
        plot_timeline(gt, pred_in, pred_ev, cfg["title"], cfg["output"],
                      ds["phase_names"], ds["colors"])


if __name__ == "__main__":
    run_curves()
    run_timelines()
    print("\nDone.")
