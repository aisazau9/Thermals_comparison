#!/bin/bash

#PBS -N rain_d03
#PBS -P up6
#PBS -q hugemem
#PBS -l walltime=02:00:00
#PBS -l ncpus=12
#PBS -l mem=500gb
#PBS -l jobfs=2gb
#PBS -l wd
#PBS -l storage=gdata/xp65+gdata/sx70+gdata/up6+gdata/hh5+gdata/rt52+gdata/zz93+gdata/hh5+scratch/up6+gdata/w28
#PBS -M a.isaza@unsw.edu.au
#PBS -m e
#PBS -r y 
#PBS -J 0-2

module use /g/data3/xp65/public/modules
module load conda/analysis3

cd /g/data/w28/ai2733/

python get_wrf_rain.py ${PBS_ARRAY_INDEX} "d03"