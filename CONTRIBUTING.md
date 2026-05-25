# Contributing

Contributions are welcome, especially improvements to reproducibility,
documentation, tests, and examples.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-dev.txt
make core
```

## Checks

Before opening a pull request, run:

```bash
make check
```

If you change numerical behavior, include a short note explaining the expected
effect on `tP`, `fcHP`, or timing results.

## Data and Artifacts

Please do not commit large waveform archives, generated matrices, or temporary
model checkpoints. Keep large artifacts outside Git and document how to
recreate or download them.
