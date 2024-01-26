#!/bin/bash

SEED=$1
curl -o model_best_new.pth -L https://api.wandb.ai/files/ulaelfray/navigation5x5_slate/7361kt0t/checkpoints/model_best.pth && \
  python dreamer.py \
    --video_pred_log False \
    --configs shapes2d \
    --task shapes2d_Navigation5x5-v0 \
    --size "64,64" \
    --ocr_config_path ocr/slate/config/navigation5x5.yaml \
    --env_config_path envs/config/navigation5x5.yaml \
    --ocr_checkpoint_path model_best_new.pth \
    --precision 16 \
    --envs 16 \
    --parallel True \
    --batch_size 8 \
    --batch_length 36 \
    --wandb_project dreamerv3-torch \
    --wandb_group navigation5x5-slots-ordered \
    --wandb_run_name run-"$SEED" \
    --seed $SEED \
    --logdir ./logdir/Navigation5x5/slots-ordered_"$SEED"
