#!/bin/bash

for i in `seq 0 7`;
do
    sbatch rebase17.slurm $i
done