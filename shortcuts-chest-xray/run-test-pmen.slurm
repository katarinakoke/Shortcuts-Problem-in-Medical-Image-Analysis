#!/bin/bash

#SBATCH --job-name=JupyterServer     # Job name
#SBATCH --output=chtrain.job.%j.out  # Name of output file (%j expands to jobID)
#SBATCH --cpus-per-task=1            # Schedule one core
#SBATCH --time=00:20:00              # Run time (hh:mm:ss)
#SBATCH --mem=100G                    # Memory requirement
#SBATCH --partition=brown            # Use the brown partition
#SBATCH --gres=gpu:a30:2             # Request one A100 40GB GPU
#SBATCH -N 1                         # Number of nodes

echo "Running on $(hostname):"

module load Anaconda3

source /home/pmen/.bashrc

echo "Activating chexpert environment..."
conda activate chexpert

/home/pmen/.conda/envs/chexpert/bin/python -u bin/test_model.py --device 0 --num_workers 2 --model_path /home/pmen/Shortcuts-Problem-in-Medical-Image-Analysis/shortcuts-chest-xray/logdirs/logdir-30k-2 --in_csv_path /home/pmen/Shortcuts-Problem-in-Medical-Image-Analysis/shortcuts-chest-xray/preprocess/datasets/random_dataset_test.csv --cfg_path /home/pmen/Shortcuts-Problem-in-Medical-Image-Analysis/shortcuts-chest-xray/config/config_pmen.json --cfg_pred /home/pmen/Shortcuts-Problem-in-Medical-Image-Analysis/shortcuts-chest-xray/predictions/