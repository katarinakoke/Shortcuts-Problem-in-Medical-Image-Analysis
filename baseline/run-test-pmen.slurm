#!/bin/bash

#SBATCH --job-name=JupyterServer     # Job name
#SBATCH --output=chtrain.job.%j.out  # Name of output file (%j $#SBATCH --cpus-per-task=1            # Schedule one core
#SBATCH --time=08:00:00              # Run time (hh:mm:ss) - 8 $#SBATCH --mem=64G                    # Memory requirement
#SBATCH --partition=brown            # Use the brown partition
#SBATCH --gres=gpu:a100_40gb:1       # Request one A100 40GB GPU
#SBATCH -N 1 # number of nodes

echo "Running on $(hostname):"

# Load the conda environment

module load Anaconda3/2021.05

# sourcing our .bashrc
source /home/pmen/.bashrc

conda activate chexpert

# Run script
python -u bin/test_model.py --device 0 --num_workers 2 --model_path /home/pmen/Shortcuts-Problem-in-Medical-Image-Analysis/baseline/logdirs --in_csv_path /home/pmen/Shortcuts-Problem-in-Medical-Image-Analysis/baseline/config/my_test.csv --cfg_path /home/pmen/Shortcuts-Problem-in-Medical-Image-Analysis/baseline/config/
