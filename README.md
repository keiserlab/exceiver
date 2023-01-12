# exceiver

[![arXiv](https://img.shields.io/badge/arXiv-2210.14330-b31b1b.svg)](https://arxiv.org/abs/2210.14330)

A pretrained single cell gene expression language model.

## install

1. clone this repository: `git clone git@github.com:keiserlab/exceiver.git`
2. install lightweight packaging tool: `conda install flit`
3. install this repo in a new environment: `flit install -s`

## usage

see `notebooks/example.ipynb` for loading pretrained models:

```
from exceiver.models import Exceiver

model = Exceiver.load_from_checkpoint("../pretrained_models/exceiver/pretrained_TS_exceiver.ckpt")
```

## preprocessing

1. downlaod the Tabula Sapiens dataset [from figshare](https://figshare.com/articles/dataset/Tabula_Sapiens_release_1_0/14267219), specifically `TabulaSapiens.h5ad.zip` (careful this link will likely begin download: https://figshare.com/ndownloader/files/34702114)
2. run `scripts/preprocess.py`:

```
python preprocess.py --ts_path /path/to/download/TabulaSapiens.h5ad
                     --out_path /path/to/prepocessed/TabulaSapiens
```

## training

`pytorch_lightning` makes distributed training easy and CLI access to a host of hyperparameters by running `scripts/train.py`:

```
python train.py --name MODELNAME 
                --data_path /path/to/prepocessed/TabulaSapiens 
                --logs path/to/model/logs
                --frac 0.15 
                --num_layers 1 
                --nhead 4 
                --query_len 128 
                --batch_size 64 
                --min_epochs 5 
                --max_epochs 10 
                --strategy ddp 
                --gpus 0,1,2,3 
```
