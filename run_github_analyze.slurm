#!/bin/bash
#SBATCH --job-name=github_analyze
#SBATCH --output=/work/barcomb_lab/Mahdi/SoftwareComponentsComparator/logs/job_output_%j.log
#SBATCH --error=/work/barcomb_lab/Mahdi/SoftwareComponentsComparator/logs/job_error_%j.log
#SBATCH --time=1-00:00:00  # 1 day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1

# Load Python
module load python/3.12.5
module load cuda/12.6.2

# Activate virtual environment
VENV_DIR="/work/barcomb_lab/Mahdi/SoftwareComponentsComparator/senv"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r /work/barcomb_lab/Mahdi/SoftwareComponentsComparator/requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

# Log start time and environment
echo "✅ Job started at $(date)"
echo "✅ Running on host: $(hostname)"
echo "✅ CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run the analysis script
python /work/barcomb_lab/Mahdi/SoftwareComponentsComparator/02_github_analyze.py

# Log end
echo "✅ Job finished at $(date)"
