#!/bin/bash

#PBS -N grid_case3_part1_new
#PBS -P n81
#PBS -q normal
#PBS -l walltime=00:30:00
#PBS -l ncpus=12
#PBS -l mem=190gb
#PBS -l jobfs=2gb
#PBS -l wd
#PBS -l storage=gdata/sx70+gdata/up6+gdata/hh5+gdata/rt52+gdata/zz93+gdata/hh5+scratch/up6+gdata/w28
#PBS -M a.isaza@unsw.edu.au
#PBS -m e
#PBS -r y
#PBS -J 1-5
#PBS -W depend=afterok:137503858.gadi-pbs

module use /g/data3/hh5/public/modules
module load conda/analysis3

cd /g/data/up6/ai2733/Codes/thermal_tracking_case3_cropped/

python get_grid_case3_d03_new.py ${PBS_ARRAY_INDEX}
