import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "MultiBypass140"

BASE_URL = "https://s3.unistra.fr/camma_public/datasets/MultiBypass140"
PARTS = [
    # "multibypass01_corrected.zip",
    "multibypass02.zip",
    "multibypass03.zip",
    "multibypass04.zip",
    "multibypass05.zip",
]
CHUNK_SIZE = 2 ** 20


def main():
  parser = argparse.ArgumentParser(description="Download MultiBypass140 dataset (videos)")
  parser.add_argument(
      "--output-dir",
      type=str,
      default=str(DATA_DIR),
      help=f"Download destination (default: {DATA_DIR})",
  )
  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  for filename in PARTS:
    url = f"{BASE_URL}/{filename}"
    archive = output_dir / filename

    print(f"Downloading {filename} ...")
    with requests.get(url, stream=True) as r:
      r.raise_for_status()
      total_mb = int(float(r.headers.get("content-length", 0)) / 10 ** 6)
      with tqdm(unit="MB", total=total_mb) as bar, open(archive, "wb") as f:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
          bar.update(len(chunk) / 10 ** 6)
          f.write(chunk)

    print(f"Extracting {filename} ...")
    with zipfile.ZipFile(archive, "r") as z:
      z.extractall(output_dir)

    archive.unlink(missing_ok=True)
    print(f"Archive {filename} removed.")

  print(f"Done. Dataset available at: {output_dir}")


if __name__ == "__main__":
  main()
