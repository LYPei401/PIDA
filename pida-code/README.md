# PIDA: Physics-Informed Design Automation with a Novel Digital Twin

PIDA is a digital-twin framework for adaptive sensing. It first predicts
where the physical state is likely to change. It then chooses a small
acquisition (sources, receivers, or stations) for that region, and
reconstructs the full field from the reduced measurements.

This repository contains the source code for the two applications in the
paper:

| Application | Task |
| --- | --- |
| [`atmospheric/`](atmospheric/) | Reconstruct global ERA5 500 hPa temperature from sparse weather stations |
| [`geology/`](geology/) | Predict CO2 leakage and choose an adaptive seismic acquisition |

The repository contains code only. Datasets and trained checkpoints are not
included. The data preparation scripts download or organize the public
datasets used in the paper.

## Repository layout

```text
atmospheric/
  data_preparation/     download WeatherBench2/ERA5 fields, build the station dataset
  pida_ml/              full-sensor residual U-Net baseline
  pida_pred/            target-adaptive PIDA sensor selection and retraining
  run_full_workflow.py  stage launcher
geology/
  data_preparation/     organize the Kimberlina-CO2 dataset
  pida_pred/            predict the next velocity map and leakage range
  pida_logn/            PIDA-Opt-logN source/receiver selection
  interfaces/           JSON schemas for the inversion backend
  case_study/           deployment configs and reported metrics for case sim0878_t60
  run_full_workflow.py  build a backend request and run the backend
```

## Installation

Python 3.10 and PyTorch 2.0 were used.

```bash
git clone <this-repository-url>
cd <repository-folder>
python -m pip install -r requirements.txt
```

A Dockerfile is also provided:

```bash
docker build -t pida .
docker run --rm pida
```

## Atmospheric application

The workflow has five stages, all run from `atmospheric/` through
`run_full_workflow.py`:

```text
prepare-field          download weekly ERA5 500 hPa temperature (WeatherBench2)
download-station-sample / analyze-stations
                       map station locations to the ERA5 grid
prepare-sensors        sample each field at the station cells
train-full-baseline    train the full-sensor residual U-Net
target-adaptive        PIDA sensor selection + fixed-mask fine-tuning per target
```

Example (the last stage for one target week at a 30% sensor reduction):

```bash
cd atmospheric
python run_full_workflow.py target-adaptive -- \
  --data_dir "$STATION_DIR" --full_checkpoint "$BASELINE_DIR/model.pt" \
  --output_dir results/target_adaptive \
  --target_rank 2 --drop_ratio 0.30 --epochs 50 --device cuda
```

See [`atmospheric/README.md`](atmospheric/README.md) for the full
command sequence. Training needs a GPU and access to WeatherBench2.

## Geology application

The geological pipeline has three stages:

1. **PIDA-Pred** ([`geology/pida_pred/`](geology/pida_pred/)) predicts the
   next velocity map and the likely leakage range.
2. **PIDA-Opt-logN** ([`geology/pida_logn/`](geology/pida_logn/)) searches
   for the smallest receiver set that meets an SSIM threshold over that
   range.
3. **Reconstruction** runs seismic-to-velocity inversion with InversionNet
   (see below).

To see the request PIDA sends to the inversion backend:

```bash
cd geology
python run_full_workflow.py --method pida_one --target sim0878_t60 --dry-run
```

To run it with your backend:

```bash
export PIDA_INVERSION_BACKEND="python /path/to/pida_openfwi_adapter.py"
python run_full_workflow.py --method pida_one --target sim0878_t60 --device cuda
```

See [`geology/README.md`](geology/README.md) for the backend interface and
data paths.

## InversionNet

The geological experiments use
[InversionNet](https://doi.org/10.1109/TCI.2019.2956866) (Wu and Lin, 2020)
for seismic-to-velocity reconstruction. They are trained with the
[OpenFWI](https://github.com/lanl/OpenFWI) implementation on the
Kimberlina-CO2 dataset (Deng et al., 2022). InversionNet is not part of this
repository. Install OpenFWI separately and connect it to PIDA through a
small adapter script. The adapter reads PIDA's JSON request and writes a JSON
result; the formats are in [`geology/interfaces/`](geology/interfaces/).

```bibtex
@article{wu2020inversionnet,
  author  = {Yue Wu and Youzuo Lin},
  title   = {InversionNet: An Efficient and Accurate Data-Driven Full Waveform Inversion},
  journal = {IEEE Transactions on Computational Imaging},
  volume  = {6},
  pages   = {419--433},
  year    = {2020}
}

@inproceedings{deng2022openfwi,
  author    = {Chengyuan Deng and Shihang Feng and Hanchen Wang and Xitong Zhang
               and Peng Jin and Yinan Feng and Qili Zeng and Yinpeng Chen and
               Youzuo Lin},
  title     = {OpenFWI: Large-Scale Multi-Structural Benchmark Datasets for
               Seismic Full Waveform Inversion},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}
```

## License

[MIT](LICENSE).
