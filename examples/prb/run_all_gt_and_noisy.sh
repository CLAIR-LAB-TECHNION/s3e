#!/bin/bash

for pair in "3 3" "5 5" "7 7" "9 9"; do
    set -- $pair
    a=$1
    b=$2
    for i in {2..8}; do
        sbatch --requeue -w socrates run_episode_gt.sh $a $b $i
        sbatch --requeue -w socrates run_episode_gt_cond.sh $a $b $i
        for j in 0.95 0.96 0.97 0.98 0.99; do
            sbatch --requeue -w socrates run_episode_noisy.sh $a $b $i $j
            sbatch --requeue -w socrates run_episode_noisy_cond.sh $a $b $i $j
        done
    done
done
