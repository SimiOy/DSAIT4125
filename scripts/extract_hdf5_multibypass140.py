import os
import cv2
import numpy as np
import h5py
import argparse
import multiprocessing
from tqdm import tqdm


def filter_black(image):
    """Detect and crop black margins. Returns cropped image, or None if no margin found."""
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary, 15, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 19)

    mask = binary[:, 10:-10]
    rows, cols = np.where(mask != 0)

    if rows.size == 0:
        return None

    left, right = int(rows.min()), int(rows.max())
    bottom, top = int(cols.min()) + 10, int(cols.max()) + 10
    w, h = right - left, top - bottom
    if w <= 0 or h <= 0:
        return None

    return image[left:left + w, bottom:bottom + h]


def process_cutmargin(frame_bgr):
    """Resize to height=300, crop black margin, then resize to 250x250. Returns BGR uint8."""
    h, w = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (int(w / h * 300), 300))
    cropped = filter_black(resized)
    base = cropped if cropped is not None else resized
    return cv2.resize(base, (250, 250))


def extract_video(video_path, video_id, output_dir):
    out_path = os.path.join(output_dir, f"{video_id}.h5")
    if os.path.exists(out_path):
        print(f"SKIP {video_id} (already exists)")
        return

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        vidcap.release()
        print(f"ERROR {video_id}: could not read FPS")
        return

    success, first = vidcap.read()
    if not success:
        vidcap.release()
        print(f"ERROR {video_id}: could not read first frame")
        return

    H, W = first.shape[:2]

    with h5py.File(out_path, "w") as f:
        ds_raw = f.create_dataset(
            "frames",
            shape=(0, H, W, 3),
            maxshape=(None, H, W, 3),
            dtype="uint8",
            chunks=(1, H, W, 3),
            compression="gzip",
            compression_opts=1,
        )
        ds_cut = f.create_dataset(
            "frames_cutmargin",
            shape=(0, 250, 250, 3),
            maxshape=(None, 250, 250, 3),
            dtype="uint8",
            chunks=(1, 250, 250, 3),
            compression="gzip",
            compression_opts=1,
        )

        def write_frame(image, idx):
            ds_raw.resize(idx + 1, axis=0)
            ds_raw[idx] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ds_cut.resize(idx + 1, axis=0)
            ds_cut[idx] = cv2.cvtColor(process_cutmargin(image), cv2.COLOR_BGR2RGB)

        frame_idx = 0
        count = 0
        step = max(1, round(fps))

        if count % step == 0:
            write_frame(first, frame_idx)
            frame_idx += 1
        count += 1

        while True:
            success, image = vidcap.read()
            if not success:
                break
            if count % step == 0:
                write_frame(image, frame_idx)
                frame_idx += 1
            count += 1

    vidcap.release()
    print(f"{video_id}: {frame_idx} frames -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/scratch/${USER}/datasets/MultiBypass140",
        help="Root MultiBypass140 directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum number of concurrent extraction processes",
    )
    args = parser.parse_args()

    data_dir = os.path.expandvars(args.data_dir)
    output_dir = os.path.join(data_dir, "frames_hdf5")
    os.makedirs(output_dir, exist_ok=True)

    video_dirs = [
        os.path.join(data_dir, "BernBypass70", "videos"),
        os.path.join(data_dir, "StrasBypass70", "videos"),
    ]

    all_videos = []
    for video_dir in video_dirs:
        if not os.path.exists(video_dir):
            print(f"MISSING video directory: {video_dir}")
            continue

        for fname in sorted(os.listdir(video_dir)):
            if fname.endswith(".mp4"):
                video_path = os.path.join(video_dir, fname)
                video_id = os.path.splitext(fname)[0]
                all_videos.append((video_path, video_id))

    print(f"Found {len(all_videos)} videos")

    processes = []
    active = []

    for video_path, video_id in all_videos:
        p = multiprocessing.Process(
            target=extract_video,
            args=(video_path, video_id, output_dir),
        )
        p.start()
        processes.append(p)
        active.append(p)

        if len(active) >= args.workers:
            active[0].join()
            active.pop(0)

    print(f"Spawned {len(processes)} video processes...")
    for p in tqdm(processes):
        p.join()

    print("Done.")


if __name__ == "__main__":
    main()
