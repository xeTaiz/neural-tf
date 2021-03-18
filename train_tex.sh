#!/bin/bash
wandb login 2a9d65e893e7a362c9eb315f11d6826fcfb1a853
#export CUDA_LAUNCH_BLOCKING=1
python pl_train_tex.py "$@"
