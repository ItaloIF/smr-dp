# Reproducibility

This document records the minimum steps needed to reproduce the bundled demo and
prepare the repository for larger experiments.

## Environment

Recommended baseline:

- Python 3.10 or newer
- `gfortran`
- dependencies from `requirements.txt`

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
make core
```

## Demo Reproduction

Run:

```bash
python3 scripts/test_matrix.py demo --trace-index 3 --save-figures
```

Expected behavior:

- reads the bundled sample stream,
- builds a displacement-frequency matrix,
- loads `models/HP_DeepLabV3Plus/best_model.pt`,
- prints timing information and the predicted coordinate,
- writes figures under `img/`.

## Batch Timing

For local full-event timing experiments:

```bash
python3 scripts/test_matrix.py benchmark \
  --event-id 20251125180103 \
  --event-root /path/to/events \
  --metadata-csv /path/to/jp_dataset_2026v2.csv
```

The benchmark path filters traces to stations listed in the metadata CSV and
prints per-trace timing for matrix generation, inference, and post-processing.

## Lightweight Checks

Run:

```bash
make check
```

This compiles Python files and catches syntax errors. It does not validate model
accuracy or install missing dependencies.

## Archival Checklist

Before creating a public release, record:

- exact dependency versions used for the paper results,
- operating system and compiler version for `core/processing.so`,
- model checkpoint checksum,
- sample data checksum,
- source dataset access instructions,
- train/validation/test split files,
- label metadata and review protocol,
- license for software, model weights, and any redistributed data.
