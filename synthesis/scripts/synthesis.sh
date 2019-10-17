#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


LOG="output/logs/synthesis.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
#exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Synthesis of linemod driller
time ./synthesis/tools/synthesis.py \
  --points_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/LOV/models/035_power_drill/035_power_drill.xyz \
  --model_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/LOV/models/035_power_drill/textured.obj \
  --pose_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/LOV/poses/035_power_drill.txt \
  --extent_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/LOV/extest.txt \
  --nb_img 20 \
  --output_dir /home/ubuntu16/Documents/PoseCNN/PoseCNN/output/synthesis/test/ \

# oooitya data
  # time ./synthesis/tools/synthesis.py \
  # --points_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/models/ooiotya/ooiotya_s.xyz \
  # --model_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/models/ooiotya/ooiotya_s.obj \
  # --pose_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/LOV/poses/035_power_drill.txt \
  # --extent_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/LOV/extest.txt \
  # --nb_img 20 \
  # --output_dir /home/ubuntu16/Documents/PoseCNN/PoseCNN/output/synthesis/test/ \

# mtrainier
  # time ./synthesis/tools/synthesis.py \
  # --points_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/models/mtrainier/mtrainier.xyz \
  # --model_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/models/mtrainier/mtrainier.obj \
  # --pose_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/LOV/poses/035_power_drill.txt \
  # --extent_path /home/ubuntu16/Documents/PoseCNN/PoseCNN/data/LOV/extest.txt \
  # --nb_img 10 \
  # --output_dir /home/ubuntu16/Documents/PoseCNN/PoseCNN/output/synthesis/test/ \