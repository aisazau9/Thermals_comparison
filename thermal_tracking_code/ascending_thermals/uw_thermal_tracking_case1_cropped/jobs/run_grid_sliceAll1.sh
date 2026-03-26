#!/bin/bash

#PBS -N grid_part1_new
#PBS -P gb02
#PBS -q hugemem
#PBS -l walltime=02:00:00
#PBS -l ncpus=12
#PBS -l mem=500gb
#PBS -l jobfs=2gb
#PBS -l wd
#PBS -l storage=gdata/sx70+gdata/up6+gdata/hh5+gdata/rt52+gdata/zz93+gdata/hh5+scratch/up6
#PBS -M a.isaza@unsw.edu.au
#PBS -W depend=afterok:126110360.gadi-pbs
#PBS -m e
#PBS -r y 
#PBS -J 1-5
#PBS -W depend=afterok:129909715.gadi-pbs

module use /g/data3/hh5/public/modules
module load conda/analysis3

cd /g/data/up6/ai2733/Codes/thermal_tracking_case1_cropped/

python get_grid_case1_d03_new.py ${PBS_ARRAY_INDEX}
