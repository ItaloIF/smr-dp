# Data Notes

This repository is designed to be a compact companion to the paper. It includes
one sample record for demonstration, but it does not include the complete
research dataset.

## Included Data

```text
data/20260106101800s.pickle
```

This sample is used by:

```bash
python3 scripts/test_matrix.py demo --trace-index 3 --save-figures
```

## Research Dataset

The paper reports a dataset of 2380 three-component strong-motion records from
K-NET and KiK-net, split at the earthquake level. The full dataset should remain
outside Git unless a small, redistributable subset is created.

Reference label tables are archived on Zenodo:

```text
Title: Reference tP and fcHP labels with earthquake-level splits (K-NET/KiK-net)
Creator: Italo Inocente
Version DOI: https://doi.org/10.5281/zenodo.18396601
Concept DOI: https://doi.org/10.5281/zenodo.18396600
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Issued: 2026-01-28
```

The Zenodo record contains:

```text
smr_dp_train_dataset.csv
smr_dp_val_dataset.csv
smr_dp_test_dataset.csv
```

According to the Zenodo metadata, these tables list record identifiers and
reference label values for joint localization of P-wave arrival time (`tP`) and
high-pass corner frequency (`fcHP`). The tables cover 2380 three-component
records from 609 earthquakes and 853 stations from NIED K-NET and KiK-net
networks for 1996-2011.

For a public release, document:

- where the raw records can be obtained,
- any license or access conditions from the data provider,
- exact event IDs and station/channel selections,
- the train/validation/test split file,
- visual-review labels for `tP` and `fcHP`,
- preprocessing parameters used to build matrices,
- checksums for any archived metadata files.

## Recommended Local Layout

For local experiments, keep large files outside the repository and point scripts
to them with command-line arguments or environment variables:

```text
/path/to/events/<event_id>/<event_id>.pickle
/path/to/metadata/jp_dataset_2026v2.csv
```

The benchmark command accepts these paths:

```bash
python3 scripts/test_matrix.py benchmark \
  --event-id 20251125180103 \
  --event-root /path/to/events \
  --metadata-csv /path/to/jp_dataset_2026v2.csv
```

## Do Not Commit

Avoid committing:

- raw waveform archives,
- generated matrices for full datasets,
- temporary model runs,
- unpublished labels or reviewer-only material,
- local absolute-path configuration files.
