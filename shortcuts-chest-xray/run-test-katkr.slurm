#!/bin/bash

#SBATCH --job-name=JupyterServer     # Job name
#SBATCH --output=chtrain.job.%j.out  # Name of output file (%j expands to jobID)
#SBATCH --cpus-per-task=1            # Schedule one core
#SBATCH --time=08:00:00              # Run time (hh:mm:ss) - 8 hours
#SBATCH --mem=64G                    # Memory requirement
#SBATCH --partition=brown            # Use the brown partition
#SBATCH --gres=gpu:a100_40gb:1       # Request one A100 40GB GPU
#SBATCH -N 1                         # Number of nodes

echo "Running on $(hostname):"

module load Anaconda3

source /home/katkr/.bashrc

echo "Activating chexpert environment..."
conda activate chexpert

/home/katkr/.conda/envs/chexpert/bin/python -u bin/test_model.py --device 0 --num_workers 2 --model_path /home/katkr/Shortcuts-Problem-in-Medical-Image-Analysis/shortcuts-chest-xray/logdirs/logdir-30k-2 --in_csv_path /home/katkr/Shortcuts-Problem-in-Medical-Image-Analysis/shortcuts-chest-xray/config/my_test.csv --cfg_path /home/katkr/Shortcuts-Problem-in-Medical-Image-Analysis/shortcuts-chest-xray/config/config_katkr.json