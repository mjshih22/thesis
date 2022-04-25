#!/bin/bash

for i in `seq 0 7`;
do
    sbatch rebase18.slurm $i
done