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
srun python valid.py --config hybrid_class_agnostic_end2end/model_58_4.yaml \
		     --out_file single_scale_flip_greedy_knn_wo_refine_wo_fillmean.txt \
		     MODEL.GC.POOL_KERNEL_SIZE 5 \
		     MODEL.GC.MASK_CROWDS False  \
		     MODEL.GC.CC_METHOD greedy \
		     MODEL.MPN.NODE_THRESHOLD 0.1 \
		     MODEL.GC.GRAPH_TYPE knn \
		     TEST.SPLIT coco_17_full \
		     TEST.WITH_REFINE False \
		     TEST.FLIP_TEST True \
		     TEST.SCORING mean \
		     TEST.FILL_MEAN False \
		     TEST.SCALE_FACTOR '[1.0]'
