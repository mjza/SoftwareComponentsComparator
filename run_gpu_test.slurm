#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=/work/barcomb_lab/Mahdi/SoftwareComponentsComparator/logs/job_output_%j.log
#SBATCH --error=/work/barcomb_lab/Mahdi/SoftwareComponentsComparator/logs/job_error_%j.log
#SBATCH --time=1-00:00:00  # 1 day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1

####### Load Modules & Virtual Environment #######
module load python/3.12.5
module load cuda/12.6.2

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
