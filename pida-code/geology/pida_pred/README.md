# PIDA-Pred: ground-truth trajectory predictor

This module implements the first stage of the geological PIDA workflow.
`train_groundtruth_pida_pred.py` learns `V(t-2), V(t-1) -> V(t)` from velocity
labels only.  It splits data by geological case, so a held-out case such as
`sim0878` does not leak into training.  `infer_next_velocity_and_range.py`
predicts a future velocity map and compares it with an explicit no-leak (or
baseline) reference map. It writes:

- `predicted_velocity.npy`: `V_hat(t)`;
- `predicted_leakage_mask.npy`: a binary `[401, 141]` leakage-region mask;
- `predicted_leakage_region.json`: changed-pixel count, bounding box, and
  inclusive horizontal interval for PIDA-Opt.

The leakage definition is `abs(V_hat(t) - V_reference) >= delta_threshold`.
The threshold, input maps, reference map, and checkpoint path are included in
the JSON output for auditability.

It is intentionally separate from PIDA-ML: PIDA-Pred forecasts velocity-map
evolution, while PIDA-ML reconstructs a velocity map from seismic acquisition.
