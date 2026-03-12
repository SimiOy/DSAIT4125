import os
import cv2
import argparse
from tqdm import tqdm


def extract_video(video_path, video_id, output_dir):
    save_dir = os.path.join(output_dir, video_id)
    os.makedirs(save_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"{video_id}: fps={fps:.2f}")
    if abs(fps - 25) > 0.5:
        print(f"  WARNING: expected 25 fps, got {fps:.2f}")

    count = 0
    second = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            if count % round(fps) == 0:
                # 1-indexed filename to match dataset loader (frame_id+1)
                fname = f"{video_id}_{second + 1:06d}.png"
                cv2.imwrite(os.path.join(save_dir, fname), image)
                second += 1
            count += 1
    vidcap.release()
    print(f"  {second} frames extracted ({count} raw frames)")
    return second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/m2cai16",
                        help="Root M2CAI16 directory")
    args = parser.parse_args()

    frames_dir = os.path.join(args.data_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    splits = [
        ("train_dataset", "workflow_video_{:02d}.mp4", range(1, 28)),
        ("test_dataset",  "test_workflow_video_{:02d}.mp4", range(1, 15)),
    ]

    total_frames = 0
    for subdir, pattern, ids in splits:
        print(f"\n=== {subdir} ===")
        for i in tqdm(ids):
            video_name = pattern.format(i)
            video_id = video_name.replace(".mp4", "")
            video_path = os.path.join(args.data_dir, subdir, video_name)
            if not os.path.exists(video_path):
                print(f"  MISSING: {video_path}")
                continue
            n = extract_video(video_path, video_id, frames_dir)
            total_frames += n

    print(f"\nDone. Total 1fps frames extracted: {total_frames}")


if __name__ == "__main__":
    main()
