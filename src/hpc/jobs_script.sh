#!/bin/bash

# job name, will show up in squeue output
#SBATCH --job-name=jobName

# mail to which notifications will be sent 
#SBATCH --mail-user=nguyed99@zedat.fu-berlin.de

# type of email notification - BEGIN, END, FAIL, ALL
#SBATCH --mail-type=FAIL,END

# ensure that all cores are on one machine
#SBATCH --nodes=1

# number of tasks/job unit
#SBATCH --ntasks=1

# number of CPUs per task
#SBATCH --cpus-per-task=24

# memory per CPU in MB (see also --mem) 
#SBATCH --mem-per-cpu=4096

# runtime in HH:MM:SS format (DAYS-HH:MM:SS format)
#SBATCH --time=0-03:00:00

# file to which standard output will be written (%A --> jobID, %a --> arrayID)
#SBATCH --output=logFiles/simulationName_%A_%a_$(date +\%d\%m\%y\%H\%M).out

# file to which standard errors will be written (%A --> jobID, %a --> arrayID)
#SBATCH --error=logFiles/simulationName_%A_%a_$(date +\%d\%m\%y\%H\%M).err

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

# # set up environment
# python3 -m venv /scratch/nguyed99/tensor # create venv
# pip install -r requirements.txt


L=50
LOG_PATH="/scratch/nguyed99/qcp-1d/logging"
timestamp=$(date +'%Y-%m-%d-%H-%M-%S')

# functions for logging memory usage and time
log_memory_cpu_usage() {
        while true; do
                echo -n "$(date +'%Y-%m-%d %H:%M:%S') " >> "$LOG_PATH/usage_log_L_${L}_${timestamp}.txt"
                # only include the used and total memory
                free -h | awk '/^Mem:/ { print $3 "/" $2 }' >> "$LOG_PATH/usage_log_L_${L}_${timestamp}.txt"
                top -bn1 | awk '/^%Cpu/ { print "CPU: " $2 " us, " $4 " sy" }' >> "$LOG_PATH/usage_log_L_${L}_${timestamp}.txt"
                sleep 900
        done
}

log_time() {
        local start=$(date +%s)
        $@
        local end=$(date +%s)
        local runtime=$((end - start))
        echo "$(date +'%Y-%m-%d %H:%M:%S') Time taken for $@: ${runtime}s" >> "$LOG_PATH/time_log_L_${L}_${timestamp}.txt"
}

# activate virtualenv
source /scratch/nguyed99/tensor/bin/activate

 # launch Python script
log_memory_cpu_usage & 
LOG_PID=$!

log_time python3 contact_process_stat_exc.py > "$LOG_PATH/contact_process_stat_exc_L_${L}_${timestamp}.out"

kill $LOG_PID