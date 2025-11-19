#!/bin/bash

for i in {2..8}; do
    sbatch --requeue -w socrates run_episode_gt.sh $i
    sbatch --requeue -w socrates run_episode_gt_cond.sh $i
    for j in 0.95 0.96 0.97 0.98 0.99; do
        sbatch --requeue -w socrates run_episode_noisy.sh $i $j
        sbatch --requeue -w socrates run_episode_noisy_cond.sh $i $j
    done
done