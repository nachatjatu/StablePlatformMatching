#!/bin/bash
#SBATCH --job-name=exp_11
#SBATCH --array=0-87
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/exp_11/out/%x_%A_%a.out
#SBATCH --error=logs/exp_11/err/%x_%A_%a.err

source .venv/bin/activate

# Load required modules
ml gurobi/11.0.2
ml python/3.12
ml py-numpy/2.2.6
ml viz
ml py-matplotlib/3.10.3_py312
ml physics
ml py-pillow-simd/10.2.0
ml py-pyproj/3.7.2
ml py-scipy/1.16.0_py312

# cd to repo root (wherever sbatch was submitted from)
cd $SLURM_SUBMIT_DIR

# Keep all native libraries within the CPUs requested from SLURM
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional but useful for your own code
export GUROBI_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Run experiment for this array index (one array task per row of small_window_instances.csv)
python3 experiments/exp_11/exp_11.py $SLURM_ARRAY_TASK_ID
