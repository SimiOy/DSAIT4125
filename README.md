# DSAIT4125 - Surgical action recognition

## Setup

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate surgical-action-recognition
```

**On DelftBlue**

Setup:
```bash
module load miniconda3
conda config --add envs_dirs /scratch/${USER}/conda/envs
conda config --add pkgs_dirs /scratch/${USER}/conda/pkgs

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f environment.yml -p /scratch/${USER}/conda/envs/surgical-action-recognition
conda activate /scratch/${USER}/conda/envs/surgical-action-recognition
```

---

## Datasets

All datasets are downloaded to `/scratch/$USER/datasets/` on the cluster. Credentials go in `.env` (copy from `.env.example`).

> **Note:** On DelftBlue, downloads must run on a **login node** - compute nodes have no internet access. To keep a download running after logout, append `& disown -h %1`.

---

### HeiCo (Heidelberg Colorectal)

Laparoscopic videos with per-frame surgical phase labels and instrument segmentation masks for proctocolectomy, rectal resection and sigmoid resection.

Requires a [Synapse](https://www.synapse.org/) account and personal access token - set `SYNAPSE_AUTH_TOKEN` in `.env`.

```bash
python scripts/download_heico.py \
    --synapse_id syn21903917 \
    --output_dir /scratch/${USER}/datasets/HeiCo
```

---

### HeiChole

Laparoscopic cholecystectomy videos with surgical phase and instrument annotations. Requires a Synapse account - set `SYNAPSE_AUTH_TOKEN` in `.env`.

```bash
python scripts/download_heico.py \
    --synapse_id syn20676772 \
    --output_dir /scratch/${USER}/datasets/HeiChole
```
---

### CholecTrack20

20 laparoscopic cholecystectomy videos annotated for multi-class multi-tool tracking, surgical phases, and scene challenges.

Requires a Synapse account + personal access token + a dataset access key received by email after filling the [request form](https://github.com/CAMMA-public/cholectrack20). Set `SYNAPSE_AUTH_TOKEN`, `SYNAPSE_EMAIL`, and `CHOLECTRACK20_KEY` in `.env`.

```bash
python scripts/download_cholectrack20.py \
    --output-dir /scratch/${USER}/datasets/CholecTrack20
```

---

### Cholec80

80 laparoscopic cholecystectomy videos with phase and tool annotations. No account required - direct download.

```bash
python scripts/download_cholec_80.py \
    --output-dir /scratch/${USER}/datasets/Cholec80
```

---

### MultiBypass140

140 laparoscopic Roux-en-Y gastric bypass videos with surgical phase and step annotations. No account required - direct download (5 parts).

```bash
python scripts/download_multibypass140.py \
    --output-dir /scratch/${USER}/datasets/MultiBypass140
```

---

### Endoscapes2023

Endoscopic scene dataset with surgical tool and anatomy annotations for laparoscopic cholecystectomy. No account required - direct download.

```bash
python scripts/download_endoscapes.py \
    --output-dir /scratch/${USER}/datasets/Endoscapes2023
```

---

### m2cai16-workflow

41 cholecystectomy videos (27 train / 14 test) with surgical workflow annotations from University Hospital of Strasbourg and Klinikum Rechts der Isar. No account required - direct download.

```bash
python scripts/download_m2cai16.py \
    --output-dir /scratch/${USER}/datasets/m2cai16
```

---

### AutoLaparo

21 laparoscopic hysterectomy videos with surgical workflow annotations. Requires requesting access at [autolaparo.github.io](https://autolaparo.github.io/) - a personal download link is emailed upon approval.

```bash
wget -P /scratch/${USER}/datasets/AutoLaparo "http://210.3.251.30:8080/share.cgi?ssid=0CZS7kO"
```