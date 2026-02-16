# DSAIT4125 - Surgical action recognition

## Setup

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate surgical-action-recognition
```

### 2. Download the Dataset

You need a [Synapse](https://www.synapse.org/) account and a [personal access token](https://accounts.synapse.org/authenticated/personalaccesstokens):

```bash
python scripts/download_data.py --token YOUR_SYNAPSE_AUTH_TOKEN
```

This downloads the dataset into the `data/` directory.