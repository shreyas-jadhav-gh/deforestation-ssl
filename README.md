# Self-Supervised Pretraining for Deforestation Segmentation

**Self-Supervised Pretraining for Satellite Image Segmentation of Deforestation and Land-Use Change**

Shreyas Jadhav and Pratham Reure, M.Tech (AIML)
Symbiosis Institute of Technology, Pune Campus, Symbiosis International (Deemed University), Pune, India
Project for the *Machine Learning Algorithms and Advances Lab*

Can self-supervised pretraining reduce the labels needed to map forest loss from satellite images, and help a model
trained in the Brazilian Amazon work in the Congo Basin? This repository builds 10 m Sentinel-2 patches (all 13 bands,
dry-season composites for 2019 and 2024) labeled with Hansen forest loss for 2020–2023. It then fine-tunes a U-Net
(ResNet-18 encoder) from four starting points:

- random weights
- the public SSL4EO-S12 MoCo weights
- two MoCo encoders pretrained here on unlabeled regional patches

Each start is trained at five label fractions with three seeds, as a single-date model and as a two-date Siamese
change-detection model. Results are checked against independent RADD radar alerts.

## Contents

- [Key results](#key-results)
- [What is where](#what-is-where)
- [Quick start: run the trained models (about 10 minutes, CPU is fine)](#quick-start-run-the-trained-models)
- [Model files](#model-files)
- [Data](#data)
- [Setup for the full pipeline](#setup-for-the-full-pipeline)
- [Reproducing the results](#reproducing-the-results)
- [Repository layout](#repository-layout)
- [Data sources, credits and license](#data-sources-credits-and-license)

## Key results

Mean over 3 seeds; threshold tuned on Amazon validation; IoU of the "cleared" class.

| Input | Encoder start | Amazon IoU, 5% of labels | Amazon IoU, 100% | Congo area called cleared, 10% of labels (true: 1.35%) |
| --- | --- | --- | --- | --- |
| 2024 only | Random | 0.477 | 0.648 | 11.0% |
| 2024 only | SSL4EO-S12 MoCo | 0.538 | 0.664 | 2.1% |
| 2024 only | MoCo, ours (Amazon + Congo) | 0.474 | 0.653 | 6.3% |
| 2024 only | MoCo, ours (Amazon only) | 0.486 | 0.645 | 14.8% |
| 2019 + 2024 | Random | 0.819 | 0.848 | 1.0% |
| 2019 + 2024 | SSL4EO-S12 MoCo | 0.786 | 0.844 | 1.9% |
| 2019 + 2024 | MoCo, ours (Amazon + Congo) | 0.826 | 0.852 | 1.4% |

- **With one image,** large-scale pretraining (SSL4EO-S12) needs about half the labels of random weights. It also keeps
  Congo false alarms near the true rate. The small-scale MoCo encoders trained here did no better than random.
- **Adding the 2019 image** is the largest gain: with 5% of the labels it beats every single-date model trained on all
  of them.
- **The labels are noisy.** Hansen labels and RADD radar alerts agree only at IoU 0.40 (Amazon) and 0.36 (Congo).

Every run's scores are in `results/kaggle_full/results/full_all.csv` (single-date) and
`results/kaggle_cd/results/cd_all.csv` (change detection); the figures are in the `figures/` folders next to them.

## What is where

| What | Where | Size |
| --- | --- | --- |
| Source code | `src/` (this repository) | < 1 MB |
| Sample data: 16 test patches with labels | `sample_data/` (this repository) | 19 MB |
| Results: scores of all runs, figures, training curves, per-patch scores, logs | `results/kaggle_lrsearch/`, `results/kaggle_full/`, `results/kaggle_cd/` (this repository) | 20 MB |
| Notebook that ran the training jobs on Kaggle | `notebooks/kaggle_job.ipynb` (this repository) | small |
| Trained models: 35 fine-tuned U-Nets and 2 pretrained MoCo encoders | [GitHub release v1.0](https://github.com/shreyas-jadhav-gh/deforestation-ssl/releases/tag/v1.0) | 1.0 GB |
| Full dataset: 2,900 labeled and 8,000 unlabeled Sentinel-2 patches | Kaggle (see [Data](#data)) | 8.4 GB |

## Quick start: run the trained models

This needs no Earth Engine account and no GPU. It runs the trained models on the 16 sample patches.

1. **Get the code.** Either `git clone https://github.com/shreyas-jadhav-gh/deforestation-ssl.git`, or on the
   repository page click **Code → Download ZIP** and unzip it. Open a terminal (on Windows: *Anaconda Prompt*) in the
   `deforestation-ssl` folder.

2. **Create the Python environment** (needs [Anaconda or Miniconda](https://www.anaconda.com/download)):
   ```
   conda create -n defor python=3.11 -y
   conda activate defor
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
   With an NVIDIA GPU, install PyTorch with the command from [pytorch.org](https://pytorch.org) instead of the
   second-last line.

3. **Download the models.** From the [v1.0 release](https://github.com/shreyas-jadhav-gh/deforestation-ssl/releases/tag/v1.0)
   download `single_date_models.zip` and `change_detection_models.zip` into the `deforestation-ssl` folder and unzip
   them there:
   ```
   tar -xf single_date_models.zip
   tar -xf change_detection_models.zip
   ```
   (`tar` is built into Windows 10 and 11; on Linux or macOS `unzip` works too.) The files land in
   `results/kaggle_full/results/checkpoints/` and `results/kaggle_cd/results/checkpoints/`.

4. **Run the demo:**
   ```
   python src/demo.py
   ```
   It runs the seven seed-0 models trained with 10% of the labels (four single-date, three change-detection), prints
   IoU, precision and recall on the Amazon and the Congo sample patches, and saves `results/demo/demo_f0.1.png` with
   the images, the label and each model's prediction. `python src/demo.py --frac 1` does the same for the models
   trained with all labels. Sixteen patches are only a sanity check; the report's numbers are on the full test sets.

## Model files

The [v1.0 release](https://github.com/shreyas-jadhav-gh/deforestation-ssl/releases/tag/v1.0) holds three zip files.
Unzip them in the repository root; the paths inside put every file where the analysis scripts look for it.

| File | Contents | Size |
| --- | --- | --- |
| `single_date_models.zip` | 20 U-Nets that see the 2024 image: 4 encoder starts × 5 label fractions (1, 5, 10, 25, 100%), seed 0 | 530 MB |
| `change_detection_models.zip` | 15 Siamese U-Nets that see 2019 and 2024: 3 encoder starts × 5 label fractions, seed 0 | 397 MB |
| `moco_encoders.zip` | the 2 ResNet-18 encoders pretrained here with MoCo v2: `moco_both` (Amazon + Congo) and `moco_amazon` (Amazon only) | 83 MB |

File names follow `<job>_<encoder start>[_cd]_lr<learning rate>_f<label fraction>_s<seed>.pt`, for example
`full_ssl4eo_lr0.0001_f0.1_s0.pt` (single-date, SSL4EO-S12 start, 10% of labels) or
`cd_scratch_cd_lr0.0002_f1_s0.pt` (change detection, random start, all labels). Only the seed-0 models were kept;
seeds 1 and 2 were evaluated in the same jobs and their scores are in the results CSVs. The probability threshold
tuned on Amazon validation for each run is the `threshold` column of those CSVs.

Loading a model in Python, from the repository root:

```python
import sys
sys.path.insert(0, "src")
from predict_examples import load_model

model, two_dates = load_model("results/kaggle_full/results/checkpoints/full_ssl4eo_lr0.0001_f0.1_s0.pt", "cpu")
# x: float tensor (N, 13, 256, 256) = Sentinel-2 L1C values / 10000, bands B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12
# change-detection models take (N, 26, 256, 256): the 13 bands of 2024 followed by the 13 bands of 2019
prob_cleared = model(x).softmax(1)[:, 1]
```

To fine-tune from one of the MoCo encoders: `python src/train_seg.py --inits moco_both=results/kaggle_full/results/pretrain/moco_both/encoder.pt ...`

## Data

**Full dataset (Kaggle).** The patches used for every experiment, exactly as uploaded for training:

- **Labeled** ([kaggle.com/datasets/shreyas631/deforestation-ssl-labeled](https://www.kaggle.com/datasets/shreyas631/deforestation-ssl-labeled), 3.5 GB): 2,900 patches of 256 × 256 pixels at 10 m: Amazon train 1,515, validation 311,
  test 308; Congo test 766. Each `patches/<id>.npz` holds `post` (2024 composite, 13 × 256 × 256, uint16),
  `pre` (2019 composite, same shape) and `label` (256 × 256, uint8: 1 = forest in 2019 cleared during 2020–2023,
  0 = forest still standing, 255 = not scored). `manifests/patches_downloaded.csv` lists every patch with its region,
  split and pixel counts.
- **Unlabeled** ([kaggle.com/datasets/shreyas631/deforestation-ssl-unlabeled](https://www.kaggle.com/datasets/shreyas631/deforestation-ssl-unlabeled), 5 GB): 8,000 patches (4,000 Amazon, 4,000 Congo)
  used only for MoCo pretraining; each `unlabeled/<id>.npz` holds `image` (13 × 256 × 256, uint16) and `year`.

To use it on your computer, download both datasets (the **Download** button on each Kaggle page) and unzip them into a
`data/` folder in the repository root (the scripts search `data/` for `patches_downloaded.csv`). On Kaggle, attach the datasets to a notebook; the scripts also search
`/kaggle/input`.

**Sample (this repository).** `sample_data/` holds 16 of the test patches (8 Amazon, 8 Congo) in the same format, enough
for `src/demo.py`. See `sample_data/README.md`.

**Rebuilding the data from scratch** needs a Google Earth Engine account; see step 1 under
[Reproducing the results](#reproducing-the-results).

Regions: the labeled patches come from two 2° × 2° boxes, Amazon (64–62°W, 10.5–8.5°S, Rondônia, Brazil) and
Congo (24–26°E, 3–1°S, eastern DR Congo). The unlabeled patches come from two neighbouring boxes that stop 0.1° short
of them (66–64.1°W and 26.1–28°E, same latitudes). Composites: Sentinel-2 Level-1C, June–September median, clouds
masked with Cloud Score+ (cs_cdf ≥ 0.60). Train, validation and test patches are separated in space (blocks of
10 × 10 patches, with a buffer).

## Setup for the full pipeline

```
conda create -n defor python=3.11
conda activate defor
# install PyTorch for your GPU from https://pytorch.org, then:
pip install -r requirements.txt
earthengine authenticate
```

Set your own Earth Engine project ID in `PROJECT` at the top of `src/dataprep/patches.py`.

Data preparation and analysis were run on a Windows laptop, training on Kaggle's "GPU T4 x2", with Python 3.11 and
segmentation-models-pytorch 0.5.0.

## Reproducing the results

1. **Data**, on a laptop with Earth Engine access:
   ```
   python src/dataprep/patches.py grid
   python src/dataprep/patches.py download
   python src/dataprep/patches.py summary
   python src/dataprep/unlabeled.py grid
   python src/dataprep/unlabeled.py download
   python src/dataprep/alerts.py download
   python src/dataprep/pack.py
   ```
   Then upload the zips from `data/upload/` to Kaggle as datasets (or use the published ones above).

2. **Training**, in a Kaggle notebook with the datasets attached and the accelerator set to "GPU T4 x2". Import
   `notebooks/kaggle_job.ipynb` (Kaggle: **File → Import Notebook**), set `JOB` in its last cell and click
   **Save Version → Save & Run All**. Run the jobs in this order, one notebook version each:
   ```
   python -u src/run_jobs.py lrsearch   # learning-rate search for random and SSL4EO-S12 starts
   python -u src/run_jobs.py full       # MoCo pretraining, MoCo learning-rate search, single-date grids
   python -u src/run_jobs.py cd         # change detection; attach the full job's output as an input first
   ```
   On a T4 a single-date run takes about 7 minutes, a change-detection run about 12, and each MoCo pretraining
   (6,000 steps) about 90 minutes.

3. **Analysis**, on the laptop, after downloading each job's output to `results/kaggle_full/` and `results/kaggle_cd/`
   (the outputs are already in this repository, and the model files are in the release):
   ```
   python src/make_figures.py --results results/kaggle_full/results results/kaggle_cd/results
   python src/predict_examples.py --results results/kaggle_full/results --frac 0.1
   python src/predict_examples.py --results results/kaggle_cd/results --frac 0.1 --tag cd --input both
   python src/validate_alerts.py --results results/kaggle_full/results results/kaggle_cd/results --tags full cd
   python src/analyse_results.py --results results/kaggle_full/results results/kaggle_cd/results
   ```
   `predict_examples.py`, `validate_alerts.py` and `analyse_results.py` also need the labeled data and the RADD
   alerts in `data/`.

The spatial split uses seed 42. Fine-tuning uses seeds 0, 1 and 2, and the labeled subset for seed s is drawn with
seed 1000 + s.

## Repository layout

```
src/
  common.py              shared helpers: finding data, loading patches, label subsets, metrics (no PyTorch)
  train_seg.py           U-Net fine-tuning and evaluation (single-date, or --input both for change detection)
  pretrain_moco.py       MoCo v2 pretraining of the 13-band ResNet-18 encoder
  run_jobs.py            the Kaggle jobs: lrsearch, full, cd (one part per GPU)
  make_figures.py        result figures and tables
  predict_examples.py    example prediction maps
  validate_alerts.py     labels and predictions against RADD radar alerts
  analyse_results.py     same-pixel comparison and haze analysis
  demo.py                quick demo of the trained models on the sample patches
  dataprep/
    patches.py           grid, Hansen labels, spatial splits, labeled patch downloads (Earth Engine)
    unlabeled.py         unlabeled patches for pretraining
    alerts.py            RADD alerts for the test patches
    pack.py              zips the data for upload to Kaggle
notebooks/
  kaggle_job.ipynb       the Kaggle notebook that runs one job
sample_data/             16 labeled test patches for the demo
results/
  kaggle_lrsearch/       output of the 'lrsearch' job: lrsearch_all.csv, curves/, hists/, per_patch/
  kaggle_full/           output of the 'full' job: full_all.csv, figures/, curves/, hists/, per_patch/, pretrain logs
  kaggle_cd/             output of the 'cd' job: cd_all.csv, figures/, curves/, hists/, per_patch/
data/                    the full dataset (not in git; see Data)
```

In the results CSVs, columns ending in `_t` use the threshold tuned on Amazon validation, columns without it use 0.5,
and `*_ap` is the threshold-free average precision.

## Data sources, credits and license

- Contains modified Copernicus Sentinel-2 data (2019, 2024), accessed through Google Earth Engine; clouds masked with
  Cloud Score+.
- Labels: Hansen/UMD/Google/USGS/NASA Global Forest Change 2000–2024, version 1.12 (Hansen et al., 2013; CC BY 4.0).
- RADD forest-disturbance alerts, Wageningen University (Reiche et al., 2021; CC BY 4.0).
- SSL4EO-S12 MoCo ResNet-18 weights (Wang et al., 2023), distributed through TorchGeo.
- U-Net implementation: segmentation_models_pytorch.

The code is released under the MIT License (see `LICENSE`). The data keep the licenses of their sources above.
