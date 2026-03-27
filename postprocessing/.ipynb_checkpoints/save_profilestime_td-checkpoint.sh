#!/bin/bash

#PBS -N save_prof
#PBS -P up6
#PBS -q hugemem
#PBS -l walltime=04:00:00
#PBS -l ncpus=12
#PBS -l mem=500gb
#PBS -l jobfs=2gb
#PBS -l wd
#PBS -l storage=gdata/sx70+gdata/up6+gdata/hh5+gdata/rt52+gdata/zz93+gdata/hh5+scratch/up6+gdata/w28+gdata/xp65
#PBS -M a.isaza@unsw.edu.au
#PBS -m e
#PBS -r y 
#PBS -J 0-2

module use /g/data3/xp65/public/modules
module load conda/analysis3

cd /g/data/w28/ai2733/

python save_profilestime.py ${PBS_ARRAY_INDEX} "td"
