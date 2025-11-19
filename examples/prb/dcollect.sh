#!/bin/bash

# processing power
#SABATCH --job-name=col-gwh
#SBATCH --requeue
#SBATCH --cpus-per-task=2

# output files
#SBATCH --output=dcollect_%A_%a.out
#SBATCH --error=dcollect_%A_%a.err

export blenderdir=$(echo photorealistic_blocksworld/blender-2.*/)

$blenderdir/blender -noaudio --background --python prb_data_collector.py
