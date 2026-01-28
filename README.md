# Automated Strong-Motion Processing: P-wave Picking and f<sub>cHP</sub> Selection

Repository with scripts and code for strong-motion record processing, dataset generation, model training, and inference.

## Repository structure
- `core/` : core processing logic and utilities
- `src/` : model code (datasets, architectures, training, inference)
- `scripts/` : runnable scripts for processing, dataset generation, training, and inference
- `data/` : data placeholders / metadata (large datasets are not stored in this repository)
- `docker/` : container setup for reproducible runs (optional)

## Setup
> Update the commands below to match your environment and dependency files.

```bash
# Example (if requirements.txt exists)
pip install -r requirements.txt

# Example (editable install, if the project is packaged)
pip install -e .
