# Atmospheric data contract

No data are included in the repository. The preparation stages build them from
the public WeatherBench2 ERA5 store and station metadata.

The target-adaptive runner requires a station dataset directory containing:

```text
Y.npy or Y_source.npy      [time, latitude, longitude] temperature field
sensor_locations.npy       [sensor_id, latitude_index, longitude_index]
splits.json                train / validation / test index lists
```

`prepare-sensors` also writes `X.npy` and `X_mask.npy` (station values and
availability mask).

This application reports RMSE, MAE, relative RMSE, and SSIM. Each
target-adaptive run stores its metrics, selected/dropped sensor indices, and
configuration next to its checkpoint.
