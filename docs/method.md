# Method Overview

SMR-DP frames automated strong-motion processing as an image localization task.
Instead of separately choosing a P-wave arrival time (`tP`) and then choosing a
high-pass corner frequency (`fcHP`), the workflow maps each acceleration trace
to a displacement-frequency image and asks a segmentation model to identify the
joint location.

## Inputs

The inference path expects one acceleration trace with:

- sample values in the trace data array,
- sampling interval `dt`,
- a fixed low-pass corner frequency, typically 30.0 Hz,
- a candidate `fcHP` grid from 0.005 to 1.28 Hz in 0.005 Hz increments.

The bundled script reads an ObsPy-supported file and selects one trace from the
stream.

## Signal Processing

For each candidate `fcHP`, the acceleration record is processed with the same
high-level steps:

1. Remove the initial baseline offset.
2. Apply a cosine taper.
3. Apply band-pass filtering.
4. Integrate acceleration to velocity and displacement with the Fortran kernel
   in `core/lib_pro.f90`.
5. Optionally fit and remove a polynomial trend in displacement.
6. Reduce extreme displacement peaks, resample to 1024 time points, and
   normalize the row.

The output is a 2D matrix with `fcHP` candidates on one axis and time on the
other. In the paper configuration, this matrix has shape `256 x 1024`.

## Model

The default model is DeepLabV3+ with an EfficientNet-B3 encoder and one output
class. It receives a single-channel displacement-frequency matrix and returns a
probability heatmap.

The checkpoint included in this repository is:

```text
models/HP_DeepLabV3Plus/best_model.pt
```

## Post-processing

The heatmap is smoothed with a Gaussian filter, and the maximum response is used
as the predicted matrix coordinate:

```text
row -> fcHP index
col -> time index
```

The time index is converted back to seconds with:

```text
tP = col / time_res * n_samples * dt
```

The frequency index can be mapped back to the candidate grid:

```text
fcHP = 0.005 * (row + 1)
```

## Reported Paper Configuration

The associated paper reports:

- 2380 three-component records from K-NET and KiK-net,
- earthquake-level train/validation/test splitting,
- candidate `fcHP` values from 0.005 to 1.28 Hz,
- 256 x 1024 displacement-frequency matrices,
- DeepLabV3+ with EfficientNet-B3 as the best-performing architecture,
- mean localization error of 5.17 pixels,
- `tP` prediction `R^2 = 0.985` for the vertical component,
- `fcHP` prediction `R^2 = 0.825` across components.

See the paper for the complete experimental protocol and interpretation:
https://doi.org/10.1785/0320260007
