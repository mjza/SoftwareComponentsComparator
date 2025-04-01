#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=/work/barcomb_lab/Mahdi/SoftwareComponentsComparator/logs/job_output_%j.log
#SBATCH --error=/work/barcomb_lab/Mahdi/SoftwareComponentsComparator/logs/job_error_%j.log
#SBATCH --time=0-04:55:00  # 4 hours 55 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1

####### Load Modules & Virtual Environment #######
module load python/3.12.5
module load cuda/11.7

VENV_DIR="/work/barcomb_lab/Mahdi/SoftwareComponentsComparator/senv"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r /work/barcomb_lab/Mahdi/SoftwareComponentsComparator/requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

####### Run your GPU Python script ###############
python /work/barcomb_lab/Mahdi/SoftwareComponentsComparator/gpu_test.py
