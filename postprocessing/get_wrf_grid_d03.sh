#!/bin/bash

#PBS -N grid_d03
#PBS -P n81
#PBS -q hugemem
#PBS -l walltime=02:00:00
#PBS -l ncpus=12
#PBS -l mem=500gb
#PBS -l jobfs=2gb
#PBS -l wd
#PBS -l storage=gdata/sx70+gdata/up6+gdata/hh5+gdata/rt52+gdata/zz93+gdata/hh5+scratch/up6+gdata/w28
#PBS -M a.isaza@unsw.edu.au
#PBS -m e
#PBS -r y 
#PBS -J 0-2

module use /g/data3/hh5/public/modules
module load conda/analysis3-23.10

cd /g/data/w28/ai2733/

python get_wrf_grid.py ${PBS_ARRAY_INDEX} "d03"
#python get_wrf_grid.py 0 "d03"