#!/bin/bash
MPI_PATH=/opt/miniconda3/envs/petsc-env/bin/
MANDYOC_PATH=$HOME/opt/mandyoc
NUMBER_OF_CORES=6
touch FD.out
$MPI_PATH/mpiexec -n $NUMBER_OF_CORES $MANDYOC_PATH/bin/mandyoc -seed 0,4 -strain_seed 0.0,1.0 | tee FD.out
bash zipper.sh
bash clean.sh
