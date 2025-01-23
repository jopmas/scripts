#!/bin/bash
MPI_PATH=$HOME/opt/petsc/arch-label-optimized/bin
MANDYOC_PATH=$HOME/opt/mandyoc/bin/mandyoc
NUMBER_OF_CORES=20
MANDYOC_OPTIONS=-seed 0,2 -strain_seed 0.0,1.0
touch FD.out
$MPI_PATH/mpiexec -n $NUMBER_OF_CORES $MANDYOC_PATH $MANDYOC_OPTIONS | tee FD.out
DIRNAME=${PWD##*/}
zip $DIRNAME.zip interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh
zip -u $DIRNAME.zip bc_velocity_*.txt
zip -u $DIRNAME.zip density_*.txt
zip -u $DIRNAME.zip heat_*.txt
zip -u $DIRNAME.zip pressure_*.txt
zip -u $DIRNAME.zip sp_surface_global_*.txt
zip -u $DIRNAME.zip strain_*.txt
zip -u $DIRNAME.zip temperature_*.txt
zip -u $DIRNAME.zip time_*.txt
zip -u $DIRNAME.zip velocity_*.txt
zip -u $DIRNAME.zip viscosity_*.txt
zip -u $DIRNAME.zip scale_bcv.txt
zip -u $DIRNAME.zip step*.txt
zip -u $DIRNAME.zip Phi*.txt
zip -u $DIRNAME.zip dPhi*.txt
zip -u $DIRNAME.zip X_depletion*.txt
zip -u $DIRNAME.zip *.log
#rm *.log
rm vel_bc*
rm velz*
rm bc_velocity*
rm velocity*
rm step*
rm temperature*
rm density*
rm viscosity*
rm heat*
rm strain_*
rm time*
rm pressure_*
rm sp_surface_global*
rm scale_bcv.txt
rm Phi*
rm dPhi*
rm X_depletion*
