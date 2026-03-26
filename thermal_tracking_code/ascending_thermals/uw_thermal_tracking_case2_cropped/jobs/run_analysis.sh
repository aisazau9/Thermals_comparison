#!/bin/bash

#PBS -N uw_case2_d03
#PBS -P up6
##PBS -q express
##PBS -l walltime=01:00:00
#PBS -q normal
#PBS -l walltime=12:00:00
#PBS -l ncpus=8
#PBS -l mem=190gb
#PBS -l jobfs=2gb
#PBS -l wd
#PBS -l storage=gdata/sx70+gdata/up6+gdata/hh5+gdata/rt52+gdata/zz93+gdata/hh5+scratch/up6+gdata/w28+gdata/xp65
#PBS -M a.isaza@unsw.edu.au
#PBS -m a

# ARG1: start_zero,ARG2:  compute_from_scracth (0 or 1)
#ARG1=$1
#ARG2=$2

module use /g/data3/xp65/public/modules
module load conda/analysis3

cd /g/data/up6/ai2733/Codes/uw_thermal_tracking_case2_cropped/

python analysis_case2_d03.py "$ARG1" "$ARG2"
