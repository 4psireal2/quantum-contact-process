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
#SBATCH --cpus-per-task=16

# memory per CPU in MB (see also --mem) 
#SBATCH --mem-per-cpu=2048

# runtime in HH:MM:SS format (DAYS-HH:MM:SS format)
#SBATCH --time=0-03:00:00

# file to which standard output will be written (%A --> jobID, %a --> arrayID)
#SBATCH --output=logFiles/simulationName_%A_%a_$(date +\%d\%m\%y\%H\%M).out

# file to which standard errors will be written (%A --> jobID, %a --> arrayID)
#SBATCH --error=logFiles/simulationName_%A_%a_$(date +\%d\%m\%y\%H\%M).err

# job arrays
#SBATCH --array=0-1


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

log_memory_cpu_usage() {
        while true; do
                echo -n "$(date +'%Y-%m-%d %H:%M:%S') " >> usage_log.txt
                # only include the used and total memory
                free -h | awk '/^Mem:/ { print $3 "/" $2 }' >> usage_log.txt
                top -bn1 | awk '/^%Cpu/ { print "CPU: " $2 " us, " $4 " sy" }' >> usage_log.txt
                sleep 900
        done
}

log_time() {
        local start=$(date +%s)
        $@
        local end=$(date +%s)
        local runtime=$((end - start))
        echo "$(date +'%Y-%m-%d %H:%M:%S') Time taken for $@: ${runtime}s" >> time_log.txt
}

# create and activate virtualenv
python3 -m venv /scratch/nguyed99/tensor
 
# launch Python script
log_memory_cpu_usage & log_time
LOG_PID=$!

/scratch/nguyed99/tensor/bin/python3 contact_process_L_10.py

kill $LOG_PID