#!/bin/bash

for file in dprocess_iter_abl*; do
    # Your commands here
    sbatch $file
done
