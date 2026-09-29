# Geological code map

```text
external data -> PIDA-Pred -> PIDA-Opt -> JSON backend request
                                                |
                                                v
                              external InversionNet/OpenFWI installation
```

| Path | Responsibility |
| --- | --- |
| `data_preparation/` | Organize and inspect the Kimberlina-CO2 data. |
| `pida_pred/` | Predict the next velocity state and export a leakage interval. |
| `pida_logn/pida_logn.py` | Expand the range, select sources, and search receiver count. |
| `case_study/configs/` | Predicted ranges and deployment records. |
| `run_full_workflow.py` | Serialize the deployment/data contract and invoke a backend. |
| external backend | Build/train/evaluate InversionNet and load checkpoints (installed separately). |
