#!/bin/bash

for i in `seq 0 7`;
do
    sbatch rebase19.slurm $i
done