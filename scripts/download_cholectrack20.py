import argparse
import os
import sys
from pathlib import Path

import requests
import synapseclient
import synapseutils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "CholecTrack20"
VALIDATE_URL = "https://synapse-response.onrender.com/validate_access"


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

    parser = argparse.ArgumentParser(description="Download CholecTrack20 dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR),
        help=f"Download destination (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    email = os.environ.get("SYNAPSE_EMAIL")
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    access_key = os.environ.get("CHOLECTRACK20_KEY")

    missing = [k for k, v in [("SYNAPSE_EMAIL", email), ("SYNAPSE_AUTH_TOKEN", token), ("CHOLECTRACK20_KEY", access_key)] if not v]
    if missing:
        print(f"Error: missing from .env: {', '.join(missing)}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Authenticating with Synapse ...")
    syn = synapseclient.login(email=email, authToken=token)

    print("Validating access key ...")
    user_id = syn.getUserProfile()["ownerId"]
    response = requests.post(VALIDATE_URL, json={"access_key": access_key, "synapse_id": user_id})
    if response.status_code != 200:
        print(f"Access key validation failed: {response.text}")
        sys.exit(1)
    entity_id = response.json()["entity_id"]
    print(f"Access granted. Entity ID: {entity_id}")

    print(f"Downloading to {output_dir} ...")
    synapseutils.syncFromSynapse(syn, entity=entity_id, path=str(output_dir))
    print("Download complete.")


if __name__ == "__main__":
    main()
