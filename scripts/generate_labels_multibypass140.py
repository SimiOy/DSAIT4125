import os
import pickle
import argparse
import h5py
from tqdm import tqdm


def load_official_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected dict in official pickle, got {type(obj)}: {path}")
    return obj


def get_hdf5_frame_count(hdf5_path):
    if not os.path.exists(hdf5_path):
        return None
    with h5py.File(hdf5_path, "r") as hf:
        if "frames" not in hf:
            raise RuntimeError(f"'frames' dataset not found in {hdf5_path}")
        return hf["frames"].shape[0]


def convert_split(official_pkl, hdf5_dir, unique_id_start=0):
    out = {}
    next_unique_id = unique_id_start
    total_frames = 0

    for video_id in tqdm(sorted(official_pkl.keys())):
        records = official_pkl[video_id]
        hdf5_path = os.path.join(hdf5_dir, f"{video_id}.h5")

        if not os.path.exists(hdf5_path):
            print(f"  MISSING HDF5: {hdf5_path} - skipping video")
            continue

        actual_frame_count = get_hdf5_frame_count(hdf5_path)
        if actual_frame_count is None or actual_frame_count == 0:
            print(f"  WARNING: no frames in {hdf5_path} - skipping video")
            continue

        frame_infos = []
        for rec in records:
            frame_id = int(rec["Original_frame_id"])

            if frame_id < 0 or frame_id >= actual_frame_count:
                print(
                    f"  Warning: {video_id} frame_id {frame_id} out of range "
                    f"(frames={actual_frame_count}), skipping"
                )
                continue

            unique_id = rec.get("unique_id", next_unique_id)
            if "unique_id" not in rec:
                next_unique_id += 1

            frame_infos.append(
                {
                    "unique_id": unique_id,
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "tool_gt": None,
                    "phase_gt": int(rec["Phase_gt"]),
                    "phase_name": None,
                    "fps": 1,
                    "frames": actual_frame_count,
                    "step_gt": int(rec["Step_gt"]) if "Step_gt" in rec else None,
                    "frame_name": rec.get("Frame_id", None),
                }
            )

        frame_infos.sort(key=lambda x: x["frame_id"])

        if len(frame_infos) == 0:
            print(f"  WARNING: no valid frame records for {video_id}, skipping video")
            continue

        out[video_id] = frame_infos
        total_frames += len(frame_infos)

    return out, total_frames, next_unique_id


def resolve_default_paths(official_labels_root, center, fold):
    base = os.path.join(
        official_labels_root,
        center,
        "labels_by70_splits",
        "labels",
    )
    train_pkl = os.path.join(base, "train", f"1fps_100_{fold}.pickle")
    val_pkl = os.path.join(base, "val", f"1fps_{fold}.pickle")
    test_pkl = os.path.join(base, "test", f"1fps_{fold}.pickle")
    return train_pkl, val_pkl, test_pkl


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/scratch/${USER}/datasets/MultiBypass140",
        help="Dataset root containing frames_hdf5/",
    )
    parser.add_argument(
        "--official-labels-root",
        required=True,
        help="Path to official labels root (the directory that contains bern/ and strasbourg/)",
    )
    parser.add_argument(
        "--center",
        default="bern",
        choices=["bern", "strasbourg"],
        help="Single-center setup to convert",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index (default: 0)",
    )
    parser.add_argument("--train-pickle", default=None)
    parser.add_argument("--val-pickle", default=None)
    parser.add_argument("--test-pickle", default=None)

    args = parser.parse_args()

    data_dir = os.path.expandvars(args.data_dir)
    official_labels_root = os.path.expandvars(args.official_labels_root)
    hdf5_dir = os.path.join(data_dir, "frames_hdf5")

    if not os.path.exists(hdf5_dir):
        raise RuntimeError(f"frames_hdf5 directory not found: {hdf5_dir}")

    default_train, default_val, default_test = resolve_default_paths(
        official_labels_root, args.center, args.fold
    )

    train_pickle = args.train_pickle or default_train
    val_pickle = args.val_pickle or default_val
    test_pickle = args.test_pickle or default_test

    print("Using official pickles:")
    print("  train:", train_pickle)
    print("  val:  ", val_pickle)
    print("  test: ", test_pickle)

    if not os.path.exists(train_pickle):
        raise RuntimeError(f"Train pickle not found: {train_pickle}")
    if not os.path.exists(val_pickle):
        raise RuntimeError(f"Val pickle not found: {val_pickle}")
    if not os.path.exists(test_pickle):
        raise RuntimeError(f"Test pickle not found: {test_pickle}")

    print("\n=== Loading official split pickles ===")
    official_train = load_official_pickle(train_pickle)
    official_val = load_official_pickle(val_pickle)
    official_test = load_official_pickle(test_pickle)

    print(f"Official train videos: {len(official_train)}")
    print(f"Official val videos:   {len(official_val)}")
    print(f"Official test videos:  {len(official_test)}")

    print("\n=== Converting TRAIN split ===")
    train_out, n_train, next_uid = convert_split(
        official_train,
        hdf5_dir,
        unique_id_start=0,
    )

    print("\n=== Converting VAL split ===")
    val_out, n_val, next_uid = convert_split(
        official_val,
        hdf5_dir,
        unique_id_start=next_uid,
    )

    print("\n=== Converting TEST split ===")
    test_out, n_test, next_uid = convert_split(
        official_test,
        hdf5_dir,
        unique_id_start=next_uid,
    )

    out_train = os.path.join(data_dir, "labels", "train", "1fpstrain.pickle")
    out_val = os.path.join(data_dir, "labels", "val", "1fpsval.pickle")
    out_test = os.path.join(data_dir, "labels", "test", "1fpstest.pickle")

    print("\n=== Saving converted pickles ===")
    save_pickle(train_out, out_train)
    save_pickle(val_out, out_val)
    save_pickle(test_out, out_test)

    print("\n=== Summary ===")
    print(f"Train videos: {len(train_out)}, frames: {n_train}")
    print(f"Val   videos: {len(val_out)}, frames: {n_val}")
    print(f"Test  videos: {len(test_out)}, frames: {n_test}")
    print("Done.")


if __name__ == "__main__":
    main()
