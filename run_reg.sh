#!/bin/bash

# In `our_inv` and `p2pinv` modes we suggest to play around with `skip` in the range [0,40] and `cfg_tar` in the range [7,18].
python3 main_run.py --mode="regression" --dataset_yaml="celeba.yaml" --skip=36 --cfg_tar=15
