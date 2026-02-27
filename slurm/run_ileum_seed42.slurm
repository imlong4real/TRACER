#!/bin/bash
#SBATCH --job-name=tracer_ileum_s42
#SBATCH --output=logs/tracer_ileum_s42_%j.out
#SBATCH --error=logs/tracer_ileum_s42_%j.err
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --mail-user=lyuan13@jhmi.edu
#SBATCH --mail-type=ALL   # send email on BEGIN, END, FAIL, REQUEUE, etc.

# ===== Fail fast =====
set -euo pipefail

echo "========== JOB INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================="

# ===== Load modules explicitly for reproducibility =====
module purge
module load gcc/9.3.0
module load bzip2/1.0.8
module load libffi/3.4.6
module load mesa/23.0.3

# ===== Ensure libraries are visible to Python =====
export LD_LIBRARY_PATH=/apps/software/spack/gcc/9.3.0/bzip2/1.0.8-vo3qutr5jxlmyt2jrnhmy3bf5dslog5k/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/software/spack/gcc/9.3.0/mesa/23.0.3-*/lib:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# ===== Activate Python environment =====
source ~/venvs/hotnerd_py311/bin/activate

# ===== Log environment for reproducibility =====
echo "Python version:"
python --version

echo "Pip freeze:"
pip freeze > logs/pip_freeze_${SLURM_JOB_ID}.txt

echo "Git commit:"
git -C ~/TRACER rev-parse HEAD

# ===== Move to repo =====
cd ~/TRACER

# ===== Run TRACER =====
python tutorials/mouse_ileum/run_mouse_ileum.py --seed 42

echo "End time: $(date)"
