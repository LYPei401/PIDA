# Atmospheric PIDA application

This application reconstructs global ERA5 500 hPa temperature fields from
sparse station measurements and the previous weekly field. It is adaptive
sensing and field reconstruction, not weather forecasting.

```text
data_preparation/       WeatherBench2/ERA5 field and station-grid preparation
pida_ml/                full-sensor residual U-Net baseline
pida_pred/              target-adaptive PIDA sensor scoring and retraining
results/                outputs from training runs
run_full_workflow.py    staged workflow launcher
```

All commands are run from `atmospheric/` as
`python run_full_workflow.py <stage> -- <stage options>`. The standalone `--`
separates the launcher from the stage options. The workflow needs access to
the public WeatherBench2 data and a GPU for training.

## Stage 1 — download/prepare weekly ERA5 fields

Prepare weekly 500 hPa ERA5 temperature maps from WeatherBench2. By default,
the script reads the public WeatherBench2 Google Cloud Zarr store:

```bash
cd atmospheric
export FIELD_DIR="$PWD/data/weekly_temperature500_1970_2019"

python run_full_workflow.py prepare-field -- \
  --output_dir "$FIELD_DIR" \
  --start_year 1970 --train_years 40 --test_years 10 \
  --variable temperature --level 500 \
  --sample_weekday 0 --sample_hour 0
```

This creates:

```text
$FIELD_DIR/Y.npy
$FIELD_DIR/times.npy
$FIELD_DIR/splits.json
$FIELD_DIR/metadata.json
```

If the WeatherBench2 Zarr data are already downloaded locally, pass the local
Zarr path with `--dataset_path /path/to/era5.zarr`.

## Stage 2 — prepare station locations

The training data require a station catalog with grid columns:

```text
grid_id_721x1440
lat_idx_721x1440
lon_idx_721x1440
```

For a small public WeatherBench2/METAR sample, run:

```bash
export STATION_SAMPLE="$PWD/data/metar_sample"
export STATION_ANALYSIS="$PWD/results/station_analysis"

python run_full_workflow.py download-station-sample -- \
  --output_dir "$STATION_SAMPLE"

python run_full_workflow.py analyze-stations -- \
  --input_dir "$STATION_SAMPLE" \
  --output_dir "$STATION_ANALYSIS" \
  --lat_order ascending
```

For a full experiment, replace the sample-derived catalog with an equivalent
complete station catalog containing the same grid columns.

## Stage 3 — build the station-grid dataset

Sample each dense ERA5 field at the mapped station grid cells:

```bash
export STATION_DIR="$PWD/data/station_temperature500_1970_2019"

python run_full_workflow.py prepare-sensors -- \
  --y_dir "$FIELD_DIR" \
  --station_catalog "$STATION_ANALYSIS/station_catalog_mapped.csv" \
  --output_dir "$STATION_DIR" \
  --grid_height 721 --grid_width 1440
```

This creates the station reconstruction dataset:

```text
$STATION_DIR/X.npy
$STATION_DIR/X_mask.npy
$STATION_DIR/Y_source.npy
$STATION_DIR/sensor_locations.npy
$STATION_DIR/splits.json
```

## Stage 4 — train the full-sensor baseline

Train the full-sensor residual U-Net baseline. This checkpoint is used to
warm-start target-adaptive PIDA runs.

```bash
export BASELINE_DIR="$PWD/results/full_residual"

python run_full_workflow.py train-full-baseline -- \
  --data_dir "$STATION_DIR" \
  --output_dir "$BASELINE_DIR" \
  --epochs 50 \
  --device cuda
```

The trained baseline checkpoint is:

```text
$BASELINE_DIR/model.pt
```

## Stage 5 — run target-adaptive PIDA training and inference

Fine-tune a fixed-mask model for one target rank and sensor-drop ratio:

```bash
python run_full_workflow.py target-adaptive -- \
  --data_dir "$STATION_DIR" \
  --full_checkpoint "$BASELINE_DIR/model.pt" \
  --output_dir "$PWD/results/target_adaptive" \
  --target_rank 2 \
  --drop_ratio 0.30 \
  --epochs 50 \
  --device cuda
```

The output directory contains the trained target model, selected/dropped sensor
indices, prediction metrics, and training history:

```text
results/target_adaptive/target_002_drop_30/
```

Repeat Stage 5 over target ranks and drop ratios to reproduce the full
target-adaptive comparison. The paper evaluates test targets and selects the
best PIDA result among the configured sensor-reduction budgets.

Additional implementation details are in `docs/ATMOSPHERIC_MAIN_WORKFLOW.md`,
`docs/ATMOSPHERIC_CODE_MAP.md`, and `docs/DATA.md`.
