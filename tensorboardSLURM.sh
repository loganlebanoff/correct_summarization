#!/bin/sh

#SBATCH --ntasks=1
#SBATCH -t 04:00:00               # max runtime is 4 hours
#SBATCH -J  tensorboard_server    # name
#SBATCH -o /lustre/fs0/home/llebanoff/discourse/logs/xsum_pgmmr_singles/tb-%J.out #TODO: Where to save your output

# To run as an array job, use the following command:
# sbatch --partition=beards --array=0-0 tensorboardHam.sh
# squeue --user thpaul


set -x

source /home/llebanoff/.bash_profile #TODO: Your profile
MODEL_DIR=/lustre/fs0/home/llebanoff/discourse/logs/xsum_pgmmr_singles/eval #TODO: Your TF model directory

let ipnport=($UID)%65274
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip


tensorboard --logdir="${MODEL_DIR}" --port=$ipnport