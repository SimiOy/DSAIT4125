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

    # Ignore 10px border columns, then find nonzero pixel coordinates with numpy
    mask = binary[:, 10:-10]
    rows, cols = np.where(mask != 0)

    if rows.size == 0:
        return None

    left, right = int(rows.min()), int(rows.max())
    bottom, top  = int(cols.min()) + 10, int(cols.max()) + 10  # adjust back for stripped columns
    w, h = right - left, top - bottom
    if w <= 0 or h <= 0:
        return None

    return image[left:left + w, bottom:bottom + h]


def process_cutmargin(frame_bgr):
    """Resize to 300-height, crop black margin, resize to 250x250. Returns BGR uint8."""
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
    if abs(fps - 25) > 0.5:
        print(f"WARNING {video_id}: expected 25 fps, got {fps:.2f}")

    raw_frames = []
    cut_frames = []
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success and count % round(fps) == 0:
            # BGR -> RGB for storage (matches PIL.Image.open convention used at train time)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            raw_frames.append(rgb)
            cut = process_cutmargin(image)
            cut_frames.append(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))
        count += 1
    vidcap.release()

    raw_arr = np.stack(raw_frames).astype(np.uint8)     # (N, H, W, 3)
    cut_arr = np.stack(cut_frames).astype(np.uint8)     # (N, 250, 250, 3)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("frames",           data=raw_arr, compression="lzf")
        f.create_dataset("frames_cutmargin", data=cut_arr, compression="lzf")

    print(f"{video_id}: {len(raw_frames)} frames → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/m2cai16")
    args = parser.parse_args()

    output_dir = os.path.join(args.data_dir, "frames_hdf5")
    os.makedirs(output_dir, exist_ok=True)

    splits = [
        ("train_dataset", "workflow_video_{:02d}.mp4",      range(1, 28)),
        ("test_dataset",  "test_workflow_video_{:02d}.mp4", range(1, 15)),
    ]

    processes = []
    for subdir, pattern, ids in splits:
        for i in ids:
            video_name = pattern.format(i)
            video_path = os.path.join(args.data_dir, subdir, video_name)
            if not os.path.exists(video_path):
                print(f"MISSING: {video_path}")
                continue
            video_id = video_name.replace(".mp4", "")
            p = multiprocessing.Process(target=extract_video,
                                        args=(video_path, video_id, output_dir))
            p.start()
            processes.append(p)

    print(f"Spawned {len(processes)} video processes...")
    for p in tqdm(processes):
        p.join()
    print("Done.")


if __name__ == "__main__":
    main()
