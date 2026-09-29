# Geological PIDA application

This application predicts a CO2-leakage region and selects an adaptive seismic
acquisition. Seismic-to-velocity reconstruction uses **InversionNet**
through the OpenFWI implementation, which is installed separately and
connected through the JSON backend interface below. The Kimberlina-CO2 data
are downloaded from OpenFWI.

```text
data_preparation/    organize externally downloaded Kimberlina-CO2 data
pida_pred/           predict the next velocity state and leakage range
pida_logn/           select sources and receivers
case_study/configs/  PIDA deployment configs for case sim0878_t60
run_full_workflow.py create a request for an external inversion backend
```

## 1. Run the PIDA-owned stages

PIDA-Pred produces a predicted leakage interval. PIDA-Opt-logN consumes that
interval and either calls an external evaluator or replays measured SSIM
values. See the component READMEs and
[docs/GEOLOGY_MAIN_WORKFLOW.md](docs/GEOLOGY_MAIN_WORKFLOW.md).

## 2. Connect an external InversionNet backend

Set a command prefix. The runner invokes it without a shell and appends
`--request-json PATH`:

```bash
export PIDA_INVERSION_BACKEND="python /outside/this/repo/pida_openfwi_adapter.py"
export PIDA_TRAIN_DATA_ROOT=/outside/this/repo/kimberlina/train/data
export PIDA_TRAIN_LABEL_ROOT=/outside/this/repo/kimberlina/train/labels
export PIDA_TEST_DATA_ROOT=/outside/this/repo/kimberlina/test/data
export PIDA_TEST_LABEL_ROOT=/outside/this/repo/kimberlina/test/labels
export PIDA_ANNO_PATH=/outside/this/repo/forwardallDM.csv

python run_full_workflow.py \
  --method pida_one \
  --target sim0878_t60 \
  --device cuda
```

To inspect the exact request without installing a backend:

```bash
python run_full_workflow.py \
  --method pida_one \
  --target sim0878_t60 \
  --dry-run
```

The backend must accept one option:

```text
--request-json /absolute/path/to/backend_request.json
```

It must write `backend_result.json` to the request's `output_dir`:

```json
{
  "schema": "pida.inversion-backend-result",
  "schema_version": 1,
  "status": "ok",
  "checkpoint": "/outside/this/repo/model.pth",
  "prediction": "/outside/this/repo/reconstruction.npy",
  "metrics": {"ssim": 0.95, "mae": 12.3, "mse": 456.7}
}
```

The backend owns model construction, OpenFWI preprocessing, training,
checkpoint loading, and inference.
PIDA owns the deployment passed in the request.

Machine-readable definitions are in
[`interfaces/backend_request.schema.json`](interfaces/backend_request.schema.json)
and
[`interfaces/backend_result.schema.json`](interfaces/backend_result.schema.json).

## Case-study replay

Supply the case data and a trained checkpoint:

```bash
python case_study/scripts/run_case_study.py \
  --input-data /outside/this/repo/sim0878_t60.npz \
  --checkpoint-path /outside/this/repo/checkpoint.pth \
  --backend-command "python /outside/this/repo/pida_openfwi_adapter.py" \
  --device cpu
```

## InversionNet

The reconstruction stage uses InversionNet (Wu and Lin, 2020) through the
OpenFWI implementation. See the [top-level README](../README.md#inversionnet)
for citations.
