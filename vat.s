#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH -t24:00:00
#SBATCH --mem=40GB
source ../../pyenv/py3.6.3/bin/activate
python train.py
