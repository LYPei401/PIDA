# PIDA-Opt-logN deployment selection

`pida_logn.py` expands the predicted leakage interval, chooses source channels,
and binary-searches the smallest receiver count meeting an SSIM threshold.

Candidate SSIM values may come from a JSON map:

```bash
python pida_logn/pida_logn.py \
  --range 88 237 \
  --ssim-threshold 0.95 \
  --candidate-ssim-json /path/to/measured_ssim.json \
  --output-json results/deployment.json
```

Alternatively, `--evaluator-command` accepts a template containing
`{receiver_count}`, `{range_start}`, `{range_end}`, `{source_count}`, or
`{source_indices}`. The command's last stdout line must be a number or JSON
object containing `ssim`. This is the intended hook for an external
InversionNet backend; no inverse-model implementation is bundled here.
