# RQ-SAIL

Official implementation of **RQ-SAIL (Risk-aware and Qualification-gated
Safety-Assured Incremental Learning)** for battery state-of-health estimation
under untrusted target-domain updates.

This repository provides the core RQ-SAIL implementation and the main NASA
BT1/BT2 experimental configurations.

## Requirements

- Python 3.10
- PyTorch 2.9.1+cu128
- CUDA 12.8

```bash
pip install -r requirements.txt
```

## Data

The released experiments use the NASA PCoE battery-aging dataset.

Place the required files under:

```text
data/raw/nasa/
├── B0005.mat
├── B0006.mat
├── B0007.mat
└── B0018.mat
```

See `data/README.md` for the battery roles and data placement.

## Running

The public workflow is:

```text
prepare -> task1 -> poison -> update -> evaluate
```

BT1:

```bash
python run_nasa.py --config configs/nasa_bt1.yaml --stage all
```

BT2:

```bash
python run_nasa.py --config configs/nasa_bt2.yaml --stage all
```

Individual stages can also be run through `--stage prepare`, `task1`,
`poison`, `update`, or `evaluate`.


