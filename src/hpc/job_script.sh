#!/bin/bash

# job name, will show up in squeue output
#SBATCH --job-name=jobName

# mail to which notifications will be sent 
#SBATCH --mail-user=nguyed99@zedat.fu-berlin.de

# type of email notification - BEGIN, END, FAIL, ALL
#SBATCH --mail-type=FAIL,END

# ensure that all cores are on one machine
#SBATCH --nodes=1

# number of job units
# SBATCH --ntasks=1

# number of CPUs per task
# SBATCH --cpus-per-task=16

# memory per CPU in MB (see also --mem) 
#SBATCH --mem-per-cpu=2048

# runtime in HH:MM:SS format (DAYS-HH:MM:SS format)
#SBATCH --time=3-00:00:00

# file to which standard output will be written (%A --> jobID, %a --> arrayID)
#SBATCH --output=logFiles/simulationName_%A_%a.out

# file to which standard errors will be written (%A --> jobID, %a --> arrayID)
#SBATCH --error=logFiles/simulationName_%A_%a.err

# job arrays
# SBATCH --array=0-10
#SBATCH --array=0-150
# SBATCH --array=0,2-4,5-9

# select partition
#SBATCH --partition=main

# set specific queue/nodes
# SBATCH --reservation=bqa

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# load Python module
module load python/3.11.*

# create and activate virtualenv
python3 setup_venv.py
source $HOME/tensor/bin/activate

# launch Python script
# python3 pythonScript.py $SLURM_ARRAY_TASK_ID
python3 qcp_hpc_scikit_tt --bond-dimension 200 300