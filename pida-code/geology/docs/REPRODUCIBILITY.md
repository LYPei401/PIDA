# Reproducibility

| Check | Command |
| --- | --- |
| Check the source tree | `python case_study/scripts/check_artifact.py --mode quick` |
| Inspect backend request | `python run_full_workflow.py --method pida_one --target sim0878_t60 --dry-run` |
| Run inference with a backend | `python case_study/scripts/run_case_study.py --input-data ... --checkpoint-path ... --backend-command ...` |

For a full reproduction, record the OpenFWI revision, adapter script, and
checkpoint hash used by the backend.
