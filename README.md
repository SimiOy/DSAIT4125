# DSAIT4125 - Surgical action recognition

## Setup

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate surgical-action-recognition
```

### 2. Download the Dataset

You need a [Synapse](https://www.synapse.org/) account and a [personal access token](https://accounts.synapse.org/authenticated/personalaccesstokens).

Copy the example env file and add your token:

```bash
cp .env.example .env
```

Run the download script with the Synapse dataset ID:

```bash
python scripts/download_data.py --synapse_id syn21903927 --output_dir data/Proctocolectomy/1/ 
```

This downloads the dataset into the `data/` directory.