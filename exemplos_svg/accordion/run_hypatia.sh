#!/usr/bin/env bash
#SBATCH --mail-type=BEGIN,END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=joao.macedo.silva@usp.br # Where to send mail
#SBATCH --ntasks=90
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00 # 16 horas; poderia ser “2-” para 2 dias; máximo “8-”
#SBATCH --job-name mandyoc-jpms
#SBATCH --output slurm_%j.log
#SBATCH --error=log_error_%j.log
#SBATCH --no-requeue
module purge
module load gcc/13.2.0-gcc-8.5.0-tnbqzki
module load openmpi/5.0.3-gcc-8.5.0-no4tqjk
module load cmake/3.27.9-gcc-8.5.0-33534nt
#Setup of Mandyoc variables:
PETSC_DIR='/home/jpmacedo/opt/petsc'
PETSC_ARCH='optimized-v3.24.1-mpich'
MANDYOC='/home/jpmacedo/opt/mandyoc/bin/mandyoc'
MANDYOC_OPTIONS='-seed 0,4 -strain_seed 0.0,1.0'
#run mandyoc
mpirun -n ${SLURM_NTASKS} --map-by :OVERSUBSCRIBE ${MANDYOC} ${MANDYOC_OPTIONS}
conda activate mpy
#Creating directories for the output files
bash mv /home/jpmacedo/opt/mv-updated.sh
#Creating netdf files
julia -t 90 /home/jpmacedo/opt/convertNETCDF_v2.jl ${PWD}
julia -t 90 /home/jpmacedo/opt/LithoNETCDF_v2.jl ${PWD}
python /home/jpmacedo/opt/track_particles_v3.py ${PWD} 0
zip ${PWD##*/}.zip *.nc
#run of auxiliary scripts to zip and clean the folder
# bash zipper.sh
# bash clean.sh
