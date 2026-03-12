import os
import cv2
import argparse
import multiprocessing
from tqdm import tqdm


def extract_video(args):
    video_path, video_id, output_dir = args
    save_dir = os.path.join(output_dir, video_id)
    os.makedirs(save_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if abs(fps - 25) > 0.5:
        print(f"  WARNING {video_id}: expected 25 fps, got {fps:.2f}")

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
    print(f"  {video_id}: {second} frames extracted")
    return second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/m2cai16")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="Parallel video workers (default: cpu count)")
    args = parser.parse_args()

    frames_dir = os.path.join(args.data_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    splits = [
        ("train_dataset", "workflow_video_{:02d}.mp4",      range(1, 28)),
        ("test_dataset",  "test_workflow_video_{:02d}.mp4", range(1, 15)),
    ]

    tasks = []
    for subdir, pattern, ids in splits:
        for i in ids:
            video_name = pattern.format(i)
            video_path = os.path.join(args.data_dir, subdir, video_name)
            if not os.path.exists(video_path):
                print(f"MISSING: {video_path}")
                continue
            video_id = video_name.replace(".mp4", "")
            tasks.append((video_path, video_id, frames_dir))

    print(f"Extracting {len(tasks)} videos with {args.workers} workers...")
    with multiprocessing.Pool(processes=args.workers) as pool:
        counts = list(tqdm(pool.imap(extract_video, tasks), total=len(tasks)))

    print(f"\nDone. Total 1fps frames extracted: {sum(counts)}")


if __name__ == "__main__":
    main()
