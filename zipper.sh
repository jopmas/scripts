DIRNAME=${PWD##*/}

echo "Zipping $DIRNAME directory..."
# zip $DIRNAME.zip interfaces.txt param.txt input*_0.txt run*.sh vel*.txt run*.sh FD.out
# zip -u $DIRNAME.zip viscosity_*.txt
# zip -u $DIRNAME.zip heat_*.txt
# zip -u $DIRNAME.zip density_*.txt
# zip -u $DIRNAME.zip temperature_*.txt
# zip -u $DIRNAME.zip time_*.txt
# zip -u $DIRNAME.zip strain_*.txt
# zip -u $DIRNAME.zip pressure*.txt
# zip -u $DIRNAME.zip sp_surface_global*.txt

zip $DIRNAME'_imgs.zip' sp_surface_global*.png
zip -u $DIRNAME'_imgs.zip' viscosity_*.png
zip -u $DIRNAME'_imgs.zip' heat_*.png
zip -u $DIRNAME'_imgs.zip' density_*.png
zip -u $DIRNAME'_imgs.zip' temperature_*.png
zip -u $DIRNAME'_imgs.zip' strain_*.png
zip -u $DIRNAME'_imgs.zip' pressure_*.png
zip -u $DIRNAME'_imgs.zip' litho_new_temper_*.png
zip -u $DIRNAME'_imgs.zip' temper_mean_rm*.png
zip -u $DIRNAME'_imgs.zip' lithology*.png

zip -r $DIRNAME'_videos.zip' *.mp4

#zip -r $DIRNAME'_PDFs.zip' *.pdf

echo "Zipping completed!"


# rm viscosity_*.png heat_*.png density_*.png temperature_*.png strain_*.png pressure_*.png litho_new_temper_*.png