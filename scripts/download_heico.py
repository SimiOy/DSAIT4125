import argparse
import os
import sys
from pathlib import Path

import synapseclient
import synapseutils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def main():
    load_env()

    parser = argparse.ArgumentParser(description="Download a dataset from Synapse")
    parser.add_argument(
        "--synapse_id",
        type=str,
        help="Synapse dataset ID (e.g. syn21903917)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DATA_DIR),
        help=f"Directory to download data into (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        print("Error: SYNAPSE_AUTH_TOKEN not found. Set it in .env")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.login(authToken=token)

    print(f"Downloading {args.synapse_id} to {output_dir} ...")
    files = synapseutils.syncFromSynapse(syn, args.synapse_id, path=str(output_dir))
    print(f"Download complete. {len(files)} files downloaded.")


if __name__ == "__main__":
    main()