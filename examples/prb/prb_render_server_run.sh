#!/bin/bash

#SBATCH --job-name=render_server
#SBATCH --output=render_server.out
#SBATCH --error=render_server.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1


xvfb-run -a -s "-screen 0 1440x900x24" photorealistic_blocksworld/blender-2.83.2-linux64/blender \
    -noaudio \
    --background \
    --python prb_render_server.py
