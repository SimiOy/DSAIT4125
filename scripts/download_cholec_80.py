'''
Author: Tong Yu
Copyright (c) University of Strasbourg. All Rights Reserved.
'''

import argparse
from pathlib import Path
import tarfile

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Cholec80"

URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz"
CHUNK_SIZE = 2 ** 20


def main():
  parser = argparse.ArgumentParser(description="Download Cholec80 dataset")
  parser.add_argument(
      "--output-dir",
      type=str,
      default=str(DATA_DIR),
      help=f"Download destination (default: {DATA_DIR})",
  )
  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  archive = output_dir / "cholec80.tar.gz"

  print(f"Downloading archive to {archive} ...")
  with requests.get(URL, stream=True) as r:
      r.raise_for_status()
      total_mb = int(float(r.headers.get("content-length", 0)) / 10 ** 6)
      with tqdm(unit="MB", total=total_mb) as bar, open(archive, "wb") as f:
          for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
              bar.update(len(chunk) / 10 ** 6)
              f.write(chunk)

  print(f"Extracting to {output_dir} ...")
  with tarfile.open(archive, "r") as t:
      t.extractall(output_dir)

  archive.unlink()
  print("Archive removed.")
  print(f"Done. Dataset available at: {output_dir}")


if __name__ == "__main__":
  main()