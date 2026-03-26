#!/bin/bash

#PBS -N merge_case2_d03
#PBS -P gb02
#PBS -q hugemem
#PBS -l walltime=01:00:00
#PBS -l ncpus=4
#PBS -l mem=800gb
#PBS -l jobfs=10gb
#PBS -l wd
#PBS -l storage=gdata/sx70+gdata/up6+gdata/hh5+gdata/rt52+gdata/zz93+gdata/hh5+scratch/up6
#PBS -M a.isaza@unsw.edu.au
#PBS -m e
#PBS -W depend=afterok:129909809[].gadi-pbs

module use /g/data3/hh5/public/modules
module load conda/analysis3

cd /g/data/up6/ai2733/Codes/thermal_tracking_case2_cropped/

python merge_grids.py
