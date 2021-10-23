#!/bin/bash
#SBATCH --job-name="test_run"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=8G
#SBATCH --time=5:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python valid_hr.py --config hrnet/w32_512_adam_lr1e-3.yaml \
		     --out_file onhuman_multi_scale_flip_validation.txt \
		     TEST.NMS_KERNEL 5 \
		     TEST.DETECTION_THRESHOLD 0.1 \
		     TEST.SPLIT ochuman_valid \
		     TEST.FLIP_TEST True \
		     TEST.REFINE True \
		     TEST.SCALE_FACTOR '[2.0, 1.0, 0.5]'
