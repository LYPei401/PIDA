# Geological case study: sim0878_t60

Deployment configs and reported metrics for the Case-5 target `sim0878_t60`.

```text
configs/backend_manifest.json                  backend interface settings
configs/deployments/pida_one/sim0878_t60.json  PIDA-one deployment
configs/deployments/pida_logn/sim0878_t60.json PIDA-Opt-logN deployment
configs/predicted_ranges.csv                   PIDA-Pred leakage interval
results/replay/**/metrics.json                 reported metrics
scripts/                                       replay and checks
```

Check the source tree:

```bash
cd geology
python case_study/scripts/check_artifact.py --mode quick
```

To regenerate the reconstruction, pass the case data, a trained checkpoint,
and your InversionNet backend to `scripts/run_case_study.py`. See
[`../README.md`](../README.md).
