#!/bin/bash

SEED=$1
curl -o enc-dec.pth -L https://api.wandb.ai/files/ulaelfray/dreamerv3-torch/2n344zbr/enc-dec.pth && \
  python dreamer.py \
    --configs shapes2d \
    --task shapes2d_Navigation5x5-v0 \
    --size "64,64" \
    --precision 16 \
    --envs 16 \
    --parallel True \
    --batch_size 8 \
    --batch_length 36 \
    --pretrained_checkpoint_path enc-dec.pth \
    --wandb_project dreamerv3-torch \
    --wandb_group navigation5x5-pretrained-enc-dec \
    --wandb_run_name run-"$SEED" \
    --seed $SEED \
    --logdir ./logdir/Navigation5x5/pretrained-enc-dec_"$SEED"