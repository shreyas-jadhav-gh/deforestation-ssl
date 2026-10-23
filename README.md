# Self-Supervised Pretraining for Deforestation Segmentation

MTech project for *Machine Learning Algorithms and Advances Lab*
Shreyas Jadhav, M.Tech (AIML), Symbiosis Institute of Technology, Pune

Can self-supervised pretraining reduce the labels needed to map forest loss from satellite images, and help a model
trained in the Brazilian Amazon work in the Congo Basin? This repository builds 10 m Sentinel-2 patches (all 13 bands,
dry-season composites for 2019 and 2024) labeled with Hansen forest loss for 2020–2023. It then fine-tunes a U-Net
(ResNet-18 encoder) from four starting points:

- random weights
- the public SSL4EO-S12 MoCo weights
- two MoCo encoders pretrained here on unlabeled regional patches

Each start is trained at five label fractions with three seeds, as a single-date model and as a two-date Siamese
change-detection model. Results are checked against independent RADD radar alerts.

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
  dataprep/
    patches.py           grid, Hansen labels, spatial splits, labeled patch downloads (Earth Engine)
    unlabeled.py         unlabeled patches for pretraining
    alerts.py            RADD alerts for the test patches
    pack.py              zips the data for upload to Kaggle
data/                    patches and manifests (not in git)
results/                 downloaded Kaggle outputs, figures (not in git)
```

## Setup

```
conda create -n defor python=3.11
conda activate defor
# install PyTorch for your GPU from https://pytorch.org, then:
pip install -r requirements.txt
earthengine authenticate
```

Set your own Earth Engine project ID in `PROJECT` at the top of `src/dataprep/patches.py`.

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
   Then upload the zips from `data/upload/` to Kaggle as datasets.

2. **Training**, in a Kaggle notebook with the datasets attached and "GPU T4 x2":
   - the notebook clones this repository, using a GitHub token stored as the Kaggle secret `GITHUB_TOKEN`
   - it installs `segmentation-models-pytorch==0.5.0`
   - it runs one job per notebook version, with "Save & Run All":
     ```
     python -u src/run_jobs.py lrsearch   # learning-rate search for random and SSL4EO-S12 starts
     python -u src/run_jobs.py full       # MoCo pretraining, MoCo learning-rate search, single-date grids
     python -u src/run_jobs.py cd         # change detection; attach the full job's output as an input first
     ```

3. **Analysis**, on the laptop, after downloading each job's output to `results/kaggle_full/` and `results/kaggle_cd/`:
   ```
   python src/make_figures.py --results results/kaggle_full/results results/kaggle_cd/results
   python src/predict_examples.py --results results/kaggle_full/results --frac 0.1
   python src/predict_examples.py --results results/kaggle_cd/results --frac 0.1 --tag cd --input both
   python src/validate_alerts.py --results results/kaggle_full/results results/kaggle_cd/results --tags full cd
   python src/analyse_results.py --results results/kaggle_full/results results/kaggle_cd/results
   ```

The spatial split uses seed 42. Fine-tuning uses seeds 0, 1 and 2, and the labeled subset for seed s is drawn with
seed 1000 + s.

## Data and credits

- Contains modified Copernicus Sentinel-2 data (2019, 2024), accessed through Google Earth Engine; clouds masked with
  Cloud Score+.
- Labels: Hansen/UMD/Google/USGS/NASA Global Forest Change 2000–2024, version 1.12 (Hansen et al., 2013; CC BY 4.0).
- RADD forest-disturbance alerts, Wageningen University (Reiche et al., 2021; CC BY 4.0).
- SSL4EO-S12 MoCo ResNet-18 weights (Wang et al., 2023), distributed through TorchGeo.
- U-Net implementation: segmentation_models_pytorch.
