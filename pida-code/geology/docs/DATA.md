# Geological data contract

Download the Kimberlina-CO2 dataset from OpenFWI and keep it outside the
repository. The backend request passes these environment-derived paths:

```text
PIDA_TRAIN_DATA_ROOT
PIDA_TRAIN_LABEL_ROOT
PIDA_TEST_DATA_ROOT
PIDA_TEST_LABEL_ROOT
PIDA_ANNO_PATH
```

For a single-case replay, `--case-input` adds an absolute NPZ path to the
request. The backend validates array shape, normalization, and compatibility
with its InversionNet checkpoint.
