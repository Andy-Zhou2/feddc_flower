# FedDC Flower

## Introduction
My project aims to implement FedDC in the Flower federated learning framework, which will facilitate easier experimentation and improve reproducibility.

## Requirements
- Anaconda or Miniconda (for managing environments)

This project is tested on Linux, macOS and Windows.

## Installation

First, clone this repository. Then run:

```
conda env create -f env.yml
conda activate feddc_flower_nightly
```

Done!
Alternatively, manually install all required libraries also work. Remember to install `flwr-nightly[simulation]` instead of `flwr` if the current released version of `flwr < 1.8`

## Usage

```
python main.py --config-name=<dataset> [dataset_partition=<partition>]
```

- `<dataset>`: This can be mnist, cifar10 or cifar100.
- `<partition>`: This could be iid, D1 or D2. iid is the default.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
