# Deployment records

`pida_one/` and `pida_logn/` contain PIDA decisions only: target, monitoring
range, selected source indices, and receiver count. They deliberately contain
no checkpoint field or model path.

Pass a record to the backend with `run_full_workflow.py`. PIDA-logN
outputs can also be supplied with `--deployment-json`.
