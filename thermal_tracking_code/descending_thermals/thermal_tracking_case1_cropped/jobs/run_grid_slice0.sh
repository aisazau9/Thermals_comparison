#!/bin/bash

#PBS -N grid_part0_new
#PBS -P gb02
#PBS -q hugemem
#PBS -l walltime=02:00:00
#PBS -l ncpus=12
#PBS -l mem=500gb
#PBS -l jobfs=2gb
#PBS -l wd
#PBS -l storage=gdata/sx70+gdata/up6+gdata/hh5+gdata/rt52+gdata/zz93+gdata/hh5+scratch/up6
#PBS -M a.isaza@unsw.edu.au
#PBS -m e

module use /g/data3/hh5/public/modules
module load conda/analysis3

cd /g/data/up6/ai2733/Codes/thermal_tracking_case1_cropped/

python get_grid_case1_d03_new.py 0
