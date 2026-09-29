# Atmospheric PIDA main workflow

The atmospheric application follows the same PIDA stages as the geological case:

```text
prepare data -> PIDA-ML full baseline -> PIDA-Pred/PIDA-Opt -> target-adaptive PIDA-ML -> evaluation
```

All commands below are run from `atmospheric/`. Outputs are written under
`results/`.

## Stage 1 — create dense weekly temperature fields

Prepare weekly 500 hPa ERA5 temperature maps from WeatherBench2. This creates
`Y.npy`, `times.npy`, `splits.json`, and metadata. The paper configuration is
1970--2019, Monday 00:00 UTC, with 1970--2009 for training and 2010--2019 for
testing.

```bash
export FIELD_DIR="$PWD/data/weekly_temperature500_1970_2019"
python run_full_workflow.py prepare-field -- \
  --output_dir "$FIELD_DIR" --start_year 1970 --train_years 40 --test_years 10 \
  --variable temperature --level 500 --sample_weekday 0 --sample_hour 0
```

## Stage 2 — derive and map station locations

The station catalog maps METAR locations to WeatherBench grid cells. To create
one from downloaded METAR sample metadata, use the first two commands. For a
full experiment, supply an equivalent complete station catalog with the
`grid_id_721x1440`, `lat_idx_721x1440`, and `lon_idx_721x1440` columns.

```bash
export STATION_SAMPLE="$PWD/data/metar_sample"
export STATION_ANALYSIS="$PWD/results/station_analysis"
python run_full_workflow.py download-station-sample -- --output_dir "$STATION_SAMPLE"
python run_full_workflow.py analyze-stations -- \
  --input_dir "$STATION_SAMPLE" --output_dir "$STATION_ANALYSIS" --lat_order ascending
```

## Stage 3 — build the station-grid reconstruction dataset

This samples each dense ERA5 field at the mapped station cells. The result has
`X.npy [time, sensors]`, `X_mask.npy`, `Y_source.npy`, `sensor_locations.npy`,
and chronological train/test indices.

```bash
export STATION_DIR="$PWD/data/station_temperature500_1970_2019"
python run_full_workflow.py prepare-sensors -- \
  --y_dir "$FIELD_DIR" \
  --station_catalog "$STATION_ANALYSIS/station_catalog_mapped.csv" \
  --output_dir "$STATION_DIR" --grid_height 721 --grid_width 1440
```

## Stage 4 — train the full-sensor residual baseline (PIDA-ML)

Method B is the paper's residual U-Net. It receives the prior field, sparse
innovation, and mask; with all sensors retained it supplies the warm-start
checkpoint for target adaptation.

```bash
export BASELINE_DIR="$PWD/results/full_residual"
python run_full_workflow.py train-full-baseline -- \
  --data_dir "$STATION_DIR" --output_dir "$BASELINE_DIR" \
  --epochs 50 --device cuda
```

The checkpoint is `$BASELINE_DIR/model.pt`.

## Stage 5 — PIDA-Pred and PIDA-Opt

The released implementation uses persistence as PIDA-Pred:

```text
S_hat(t) = D(t-1)
```

PIDA-Opt scores each station using the prior field's temporal change and
spatial gradient, then drops the lowest-scoring locations for a requested
budget. The exact selector is implemented in
`pida_pred/target_adaptive_pida/run_target_adaptive.py` as `select_pida_sensors`.
Its selected/dropped arrays are saved with every target run.

## Stage 6 — target-adaptive PIDA-ML and evaluation

Fine-tune a fixed-mask U-Net for one target rank and sensor budget. This
writes the target checkpoint, selection arrays, training history, and full
baseline/PIDA metrics.

```bash
python run_full_workflow.py target-adaptive -- \
  --data_dir "$STATION_DIR" --full_checkpoint "$BASELINE_DIR/model.pt" \
  --output_dir "$PWD/results/target_adaptive" \
  --target_rank 2 --drop_ratio 0.30 --epochs 50 --device cuda
```

The result is `results/target_adaptive/target_002_drop_30/`. Repeat
over target ranks and budgets, then select the configuration with the highest
PIDA SSIM per target. The paper evaluates 50 test targets and selects the best
configuration among 10%, 20%, 30%, 40%, 50%, and 60% reductions.

## Requirements

The full 50-year workflow requires a GPU, WeatherBench2/ERA5 access, and a
station catalog. The file contract is documented in `docs/DATA.md`.
