#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=slurm-cpu-%J.out

# Give this process 1 task (per GPU, but only one GPU), then assign eight 8per task
# (so 8 cores overall).  Then enforce that slurm assigns only CPUs that are on the
# socket closest to the GPU you get.

# If you want two GPUs:
# #SBATCH --ntasks=2
# #SBATCH --gres=gpu:2


# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"


echo

# OR, instead:  Activate the non-GPU version of TensorFlow
#source activate tensorflow

echo "${1}"

# Run TensorFlow
echo
${1}
echo

# You're done!
echo "Ending script..."
date

