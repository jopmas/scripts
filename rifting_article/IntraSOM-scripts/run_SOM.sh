#!/usr/bin/env bash

#SBATCH --ntasks=96 # number of cores
#SBATCH --cpus-per-task=1 # number of cpu’s to be used (the less, the better)
#SBATCH --time 72:00:00 # timer to stop simulation
#SBATCH --mem=35G       # ajuste conforme necessário
#SBATCH --job-name mandySOM # simulation ID (should contain max. 10 characters)
#SBATCH --output slurm_%j.log # output file name
#SBATCH --mail-user=jbueno@usp.br # send email
#SBATCH --mail-type=BEGIN,END,FAIL

echo JobID: $SLURM_JOB_ID # ID of job allocation
echo JobDir: $SLURM_SUBMIT_DIR # Directory job where was submitted
echo JobNodes: $SLURM_JOB_NODELIST # File containing allocated hostnames
echo JobTasks: $SLURM_NTASKS # Total number of cores for job

#---
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
#---

module unload anaconda3
source ~/opt/miniconda3/bin/activate
conda activate intrasom_env
echo running SOM
echo command: $SLURM_SUBMIT_DIR/SOM_hypatia.py $SLURM_NTASKS
python3 $SLURM_SUBMIT_DIR/SOM_hypatia.py $SLURM_NTASKS
echo finished!
