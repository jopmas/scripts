DIRNAME=${PWD##*/}

echo "Zipping $DIRNAME directory..."
# zip $DIRNAME.zip interfaces.txt param.txt input*_0.txt run*.sh vel*.txt run*.sh FD.out
 zip -u $DIRNAME.zip viscosity_*.txt
 zip -u $DIRNAME.zip radiogenicc_heat*.txt
 zip -u $DIRNAME.zip density_*.txt
 zip -u $DIRNAME.zip temperature_*.txt
 zip -u $DIRNAME.zip time_*.txt
 zip -u $DIRNAME.zip strain_*.txt
 zip -u $DIRNAME.zip pressure*.txt
 zip -u $DIRNAME.zip sp_surface_global*.txt
 zip -u $DIRNAME.zip step*.txt

zip $DIRNAME'_imgs.zip'    _output/*surface*.png
zip -u $DIRNAME'_imgs.zip' _output/*viscosity_*.png
zip -u $DIRNAME'_imgs.zip' _output/*radiogenic_heat*.png
zip -u $DIRNAME'_imgs.zip' _output/*density_*.png
zip -u $DIRNAME'_imgs.zip' _output/*temperature_*.png
zip -u $DIRNAME'_imgs.zip' _output/*strain_*.png
zip -u $DIRNAME'_imgs.zip' _output/*pressure_*.png
zip -u $DIRNAME'_imgs.zip' _output/*temperature_anomaly*.png
zip -u $DIRNAME'_imgs.zip' _output/*lithology*.png

zip -r $DIRNAME'_videos.zip' *.mp4

zip -r $DIRNAME'_gifs.zip' *.gif

#zip -r $DIRNAME'_PDFs.zip' *.pdf

echo "Zipping completed!"

# rm viscosity_*.png heat_*.png density_*.png temperature_*.png strain_*.png pressure_*.png litho_new_temper_*.png
