# Atmospheric code map

Only the production PIDA path is retained:

```text
prepare dense fields and station grid
    -> train full-sensor residual baseline
    -> score/drop low-value sensors for each target
    -> fine-tune fixed-mask PIDA model
    -> reconstruct and evaluate the target field
```

| File or directory | Role |
| --- | --- |
| `data_preparation/prepare_wb2_weekly_range.py` | Prepare weekly WeatherBench2/ERA5 500 hPa temperature fields and chronological splits. |
| `data_preparation/download_wb2_metar_docs_sample.py`, `data_preparation/analyze_wb2_metar_stations.py` | Obtain and map station metadata to the ERA5 grid. |
| `data_preparation/prepare_station_location_sensor_xy_range.py` | Sample fields at mapped station cells to construct the sensor dataset. |
| `pida_ml/models/sensor_only_unet.py` | Residual U-Net used for both baseline and PIDA reconstruction. |
| `pida_ml/train_full_baseline.py`, `pida_ml/full_baseline.py` | Train the full-sensor residual baseline used to warm-start PIDA. |
| `pida_pred/target_adaptive_pida/run_target_adaptive.py` | Persistence prior, temporal/gradient sensor scoring, mask selection, fine-tuning, and evaluation. |
| `results/` | Outputs of training runs. |
| `run_full_workflow.py` | The supported workflow entry point. |
