# Sample data

Sixteen labeled test patches, in the same format as the full dataset, so that `python src/demo.py` runs without
downloading the 8 GB dataset.

- `patches/<id>.npz`: `post` (2024 composite) and `pre` (2019 composite), each 13 × 256 × 256 uint16 Sentinel-2 L1C
  values (bands B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12, 10 m pixels), and `label` (256 × 256 uint8:
  1 = forest in 2019 cleared during 2020–2023, 0 = forest still standing, 255 = not scored).
- `manifests/patches_downloaded.csv`: the 16 patches with region, split and pixel counts.
- `manifests/band_stats.json`: per-band statistics written during data preparation (a copy of the full dataset's file).

Which patches: the seven shown in the report's example-prediction figure (three Amazon and three Congo test patches
with clearing, and the Congo patch where the randomly initialised model raised the most false alarms), plus five
more Amazon and four more Congo test patches with clearing, drawn at random. All are from the test sets, never used
in training.
