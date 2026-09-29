# Checkpoint contract

Checkpoints are produced by the backend. Supply an absolute path when
requesting inference only:

```bash
python run_full_workflow.py \
  --method pida_one \
  --target sim0878_t60 \
  --skip-train \
  --checkpoint-path /outside/this/repo/checkpoint.pth \
  --case-input /outside/this/repo/sim0878_t60.npz \
  --backend-command "python /outside/this/repo/pida_openfwi_adapter.py"
```

The checkpoint must be compatible with the backend. Do not substitute a checkpoint trained for a
different target or deployment.
