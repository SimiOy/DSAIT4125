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

    success, first = vidcap.read()
    if not success:
        vidcap.release()
        print(f"ERROR {video_id}: could not read first frame")
        return
    H, W = first.shape[:2]

    with h5py.File(out_path, "w") as f:
        ds_raw = f.create_dataset("frames",           shape=(0, H, W, 3),
                                  maxshape=(None, H, W, 3),
                                  dtype="uint8", chunks=(1, H, W, 3), compression="gzip", compression_opts=1)
        ds_cut = f.create_dataset("frames_cutmargin", shape=(0, 250, 250, 3),
                                  maxshape=(None, 250, 250, 3),
                                  dtype="uint8", chunks=(1, 250, 250, 3), compression="gzip", compression_opts=1)

        def write_frame(image, idx):
            ds_raw.resize(idx + 1, axis=0)
            ds_raw[idx] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ds_cut.resize(idx + 1, axis=0)
            ds_cut[idx] = cv2.cvtColor(process_cutmargin(image), cv2.COLOR_BGR2RGB)

        frame_idx = 0
        count = 0
        step = round(fps)

        if count % step == 0:      # first frame already read
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
    print(f"{video_id}: {frame_idx} frames → {out_path}")


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
