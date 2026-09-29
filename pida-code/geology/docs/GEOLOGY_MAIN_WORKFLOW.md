# Geological PIDA workflow

## Stage 1 — PIDA-Pred

`pida_pred/train_groundtruth_pida_pred.py` trains the two-input velocity
predictor and `infer_next_velocity_and_range.py` exports the predicted velocity,
leakage mask, bounding box, and horizontal interval.

The full Kimberlina-CO2 dataset is external. Keep complete geological cases in
the same train/test split with
`data_preparation/split_co2_dataset_by_case.py`.

## Stage 2 — PIDA-Opt

`pida_logn/pida_logn.py` expands narrow monitoring ranges, selects source
indices, and binary-searches the minimum receiver count satisfying an SSIM
threshold. It accepts either previously measured SSIM values or an external
evaluator command whose final output line is `{"ssim": NUMBER}`.

## Stage 3 — external PIDA-ML backend

The study used InversionNet (via OpenFWI) for seismic-to-velocity
reconstruction. `run_full_workflow.py` writes a versioned request containing:

- PIDA method, target, monitoring range, sources, and receiver count;
- external train/test data paths;
- operation (`train_and_infer` or `infer`), checkpoint, and device;
- training seed and epoch settings; and
- the required result path.

The separately installed backend accepts `--request-json PATH` and writes a
versioned result with its output paths and metrics.

## Stage 4 — evidence

Reported metrics for `sim0878_t60` are under `case_study/results/`.
Reproducing them requires the input data, a checkpoint, and the backend.
