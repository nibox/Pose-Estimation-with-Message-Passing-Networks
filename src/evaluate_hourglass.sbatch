#!/bin/bash
#SBATCH --job-name="test_run"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python valid.py --config hourglass/model_70_4.yaml \
		     --out_file coco_test.txt \
		     MODEL.GC.POOL_KERNEL_SIZE 3 \
		     MODEL.GC.MASK_CROWDS False  \
		     MODEL.GC.CC_METHOD GAEC \
		     MODEL.MPN.NODE_THRESHOLD 0.1 \
		     MODEL.GC.GRAPH_TYPE knn \
		     TEST.SPLIT test-dev2017 \
		     TEST.WITH_REFINE True \
		     TEST.FLIP_TEST True \
		     TEST.SCORING correct \
		     TEST.FILL_MEAN False \
		     TEST.SCALE_FACTOR '[2.0, 1.0, 0.5]'
