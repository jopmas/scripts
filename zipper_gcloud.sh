DIRNAME=${PWD##*/}

echo "Zipping $DIRNAME directory..."
zip $DIRNAME.zip interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh FD.out
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

echo "Zipping completed!"
