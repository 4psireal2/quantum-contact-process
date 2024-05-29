#!/bin/bash

# job name, will show up in squeue output
#SBATCH --job-name=cp-stat

# mail to which notifications will be sent 
#SBATCH --mail-user=nguyed99@zedat.fu-berlin.de

# type of email notification - BEGIN, END, FAIL, ALL
#SBATCH --mail-type=FAIL,END

# ensure that all cores are on one machine
#SBATCH --nodes=1

# number of tasks/job unit
#SBATCH --ntasks=1

# number of CPUs per task
#SBATCH --cpus-per-task=26

# memory per CPU in MB (see also --mem) 
#SBATCH --mem-per-cpu=4096

# runtime in HH:MM:SS format (DAYS-HH:MM:SS format)
#SBATCH --time=6-00:00:00

# file to which standard output will be written (%A --> jobID, %a --> arrayID)
#SBATCH --output=/scratch/nguyed99/qcp-1d/logging/cp_stat_exc_%A_%a.out

# file to which standard errors will be written (%A --> jobID, %a --> arrayID)
#SBATCH --error=/scratch/nguyed99/qcp-1d/logging/cp_stat_exc_%A_%a.err

# job arrays
#SBATCH --array=0-25

# select partition
#SBATCH --partition=main

# set specific queue/nodes
# SBATCH --reservation=bqa

# simulation parameter
L=50
OMEGAS=(0.0 0.9 1.8 2.7 3.6 4.5 5.4 6.3 7.2 8.1 9.0 9.9 10.8)
BOND_DIMS=(35 50)
OMEGA_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
BOND_DIM_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
OMEGA=${OMEGAS[OMEGA_INDEX]}
BOND_DIM=${BOND_DIMS[BOND_DIM_INDEX]}

# paths and file names
timestamp=$(date +'%Y-%m-%d-%H-%M-%S')
LOG_PATH="/scratch/nguyed99/qcp-1d/logging"

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID
echo "slurm task ID = $SLURM_ARRAY_TASK_ID"
echo $OMEGA $BOND_DIM

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# # set up environment
# python3 -m venv /scratch/nguyed99/tensor # create venv
# pip install -r requirements.txt

# activate virtualenv
source /scratch/nguyed99/tensor/bin/activate

export PYTHONPATH=$PYTHONPATH:/scratch/nguyed99/qcp-1d
echo "Output log" >> "$LOG_PATH/${timestamp}.log"
python3 contact_process_stat.py $L $OMEGA $BOND_DIM $SLURM_ARRAY_JOB_ID 2>&1 | awk -v task_id=$SLURM_ARRAY_TASK_ID '{print "array task " task_id, $0}' >> "$LOG_PATH/${timestamp}.log"