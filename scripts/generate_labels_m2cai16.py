import os
import pickle
import argparse
from tqdm import tqdm


PHASE2ID = {
    'TrocarPlacement':        0,
    'Preparation':            1,
    'CalotTriangleDissection': 2,
    'ClippingCutting':        3,
    'GallbladderDissection':  4,
    'GallbladderPackaging':   5,
    'CleaningCoagulation':    6,
    'GallbladderRetraction':  7,
}


def load_annotation(ann_path):
    """Return dict: raw_25fps_frame_index -> phase_name."""
    phase_dict = {}
    with open(ann_path, 'r') as f:
        lines = f.readlines()[1:]  # skip header
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            phase_dict[int(parts[0])] = parts[1]
    return phase_dict


def build_split(split_dir, pattern, video_ids, frames_dir):
    """Build the pickle dict for one split.

    Returns (pkl_dict, total_frames_count).
    """
    pkl = {}
    unique_id = 0

    for i in tqdm(video_ids):
        video_name = pattern.format(i)
        video_id = video_name.replace(".txt", "")
        ann_path = os.path.join(split_dir, video_name)

        if not os.path.exists(ann_path):
            print(f"  MISSING annotation: {ann_path}")
            continue

        video_frames_dir = os.path.join(frames_dir, video_id)
        if not os.path.isdir(video_frames_dir):
            print(f"  MISSING frames dir: {video_frames_dir} — skipping")
            continue

        # Count actual extracted frames (source of truth)
        frame_files = sorted(f for f in os.listdir(video_frames_dir) if f.endswith('.png'))
        actual_frame_count = len(frame_files)
        if actual_frame_count == 0:
            print(f"  WARNING: no frames found in {video_frames_dir}")
            continue

        phase_dict = load_annotation(ann_path)

        frame_infos = []
        for frame_id_1fps in range(actual_frame_count):
            raw_frame_id = frame_id_1fps * 25
            phase_name = phase_dict.get(raw_frame_id)
            if phase_name is None:
                print(f"  Warning: no annotation for {video_id} 1fps-frame {frame_id_1fps} "
                      f"(raw {raw_frame_id}), skipping")
                continue
            if phase_name not in PHASE2ID:
                print(f"  Warning: unknown phase '{phase_name}' in {video_id}, skipping")
                continue

            frame_infos.append({
                'unique_id':  unique_id,
                'frame_id':   frame_id_1fps,   # 0-indexed; file = {video_id}_{frame_id+1:06d}.png
                'video_id':   video_id,
                'tool_gt':    None,
                'phase_gt':   PHASE2ID[phase_name],
                'phase_name': phase_name,
                'fps':        1,
                'frames':     actual_frame_count,
            })
            unique_id += 1

        pkl[video_id] = frame_infos

    return pkl, unique_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/m2cai16",
                        help="Root M2CAI16 directory")
    args = parser.parse_args()

    frames_dir = os.path.join(args.data_dir, "frames")

    print("=== Building TRAIN pickle ===")
    train_pkl, n_train = build_split(
        split_dir=os.path.join(args.data_dir, "train_dataset"),
        pattern="workflow_video_{:02d}.txt",
        video_ids=range(1, 28),
        frames_dir=frames_dir,
    )

    print("\n=== Building TEST pickle ===")
    test_pkl, n_test = build_split(
        split_dir=os.path.join(args.data_dir, "test_dataset"),
        pattern="test_workflow_video_{:02d}.txt",
        video_ids=range(1, 15),
        frames_dir=frames_dir,
    )

    train_out = os.path.join(args.data_dir, "labels", "train")
    os.makedirs(train_out, exist_ok=True)
    with open(os.path.join(train_out, "1fpstrain.pickle"), "wb") as f:
        pickle.dump(train_pkl, f)

    test_out = os.path.join(args.data_dir, "labels", "test")
    os.makedirs(test_out, exist_ok=True)
    with open(os.path.join(test_out, "1fpstest.pickle"), "wb") as f:
        pickle.dump(test_pkl, f)

    print(f"\nTrain videos: {len(train_pkl)},  frames: {n_train}")
    print(f"Test  videos: {len(test_pkl)},  frames: {n_test}")
    print("Done.")


if __name__ == "__main__":
    main()
