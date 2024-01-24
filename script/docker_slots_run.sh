#!/bin/bash

SEED=$1
curl -o model_best.pth -L https://api.wandb.ai/files/ulaelfray/navigation7x7_slate/pcpd0frz/checkpoints/model_best.pth && \
  python dreamer_slots.py \
          --configs shapes2d-slots \
          --task shapes2d_Pushing7x7-v0 \
          --size "64,64" \
          --ocr_config_path ocr/slate/config/navigation5x5.yaml \
          --env_config_path envs/config/navigation5x5.yaml \
          --ocr_checkpoint_path model_best.pth \
          --precision 16 \
          --envs 16 \
          --parallel True \
          --dyn_deter 512 \
          --dyn_hidden 512 \
          --batch_size 8 \
          --batch_length 36 \
          --wandb_project dreamerv3-torch_pushing7x7 \
          --wandb_run_name run-"$SEED" \
          --wandb_group slots \
          --seed "$SEED" \
          --logdir ./logdir/slots_"$SEED"
