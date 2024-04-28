#!/bin/bash

#SBATCH --job-name=JupyterServer     # Job name
#SBATCH --output=chtrain.job.%j.out  # Name of output file (%j expands to jobID)
#SBATCH --cpus-per-task=1            # Schedule one core
#SBATCH --time=09:00:00              # Run time (hh:mm:ss) - 8 hours
#SBATCH --mem=55G                    # Memory requirement
#SBATCH --partition=brown            # Use the brown partition
#SBATCH --gres=gpu:a30:4             # Request one A100 40GB GPU
#SBATCH -N 1                         # Number of nodes

echo "Running on $(hostname):"

module load Anaconda3

source /home/katkr/.bashrc

echo "Activating chexpert environment..."
conda activate chexpert

/home/katkr/.conda/envs/chexpert/bin/python -u bin/train.py /home/katkr/Shortcuts-Problem-in-Medical-Image-Analysis/shortcuts-chest-xray/config/config_katkr.json /home/katkr/Shortcuts-Problem-in-Medical-Image-Analysis/shortcuts-chest-xray/logdirs/logdir-30k-2 --device 0 --num_workers 2 --verbose true