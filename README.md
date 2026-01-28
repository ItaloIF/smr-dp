# smr-dp

Automated strong-motion record processing using deep learning, focused on **simultaneous P-wave arrival identification** and **high-pass corner frequency selection (fcHP)**. :contentReference[oaicite:1]{index=1}

This repository accompanies the research:
**Automated Strong-Motion Record Processing via Deep Learning–Based Simultaneous P-Wave Identification and High-Pass Corner Frequency Selection**. :contentReference[oaicite:2]{index=2}

## What is included
End-to-end code used for:
- Strong-motion record processing workflows
- Dataset generation for learning tasks
- Model training
- Inference / prediction on new records :contentReference[oaicite:3]{index=3}

## Repository structure
Top-level folders:
- `core/` : core processing logic and utilities :contentReference[oaicite:4]{index=4}
- `src/` : model code (architectures, training, inference modules) :contentReference[oaicite:5]{index=5}
- `scripts/` : runnable scripts for processing, dataset generation, training, inference :contentReference[oaicite:6]{index=6}
- `data/` : data placeholders, examples, or metadata (see notes below) :contentReference[oaicite:7]{index=7}
- `docker/` : container setup for reproducible runs :contentReference[oaicite:8]{index=8}

> Note: Large datasets are typically not stored directly in GitHub. If the repository uses external data sources, see the `data/` folder and script/config comments for expected formats and paths.

## Quickstart

### Option A: Local Python environment
1) Create and activate an environment (example):

