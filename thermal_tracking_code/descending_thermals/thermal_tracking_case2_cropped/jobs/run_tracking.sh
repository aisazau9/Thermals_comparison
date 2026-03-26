#!/bin/bash

#PBS -N case2_d03
#PBS -P gb02
#PBS -q hugemem
#PBS -l walltime=08:00:00
#PBS -l ncpus=8
#PBS -l mem=1000gb
#PBS -l jobfs=2gb
#PBS -l wd
#PBS -l storage=gdata/sx70+gdata/up6+gdata/hh5+gdata/rt52+gdata/zz93+gdata/hh5+scratch/up6
#PBS -M a.isaza@unsw.edu.au
#PBS -m e
#PBS -W depend=afterok:126734694.gadi-pbs

module use /g/data3/hh5/public/modules
module load conda/analysis3

cd /g/data/up6/ai2733/Codes/thermal_tracking_case2_cropped/

python tracking_runscript_case2_d03.py
