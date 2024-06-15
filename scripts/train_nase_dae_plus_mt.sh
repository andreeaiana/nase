#!/bin/bash
source $HOME/.bashrc
conda activate nase_env

python nase/train.py experiment=train_nase_dae_plus_mt