#!/bin/bash
DIRNAME=${PWD##*/}
# ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'viscosity_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_viscosity.mp4"
# ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'heat_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_heat.mp4"
# ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'density_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_density.mp4"
# ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'strain_[0-9]*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_strain.mp4"
ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'strain_rate*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_strain_rate.mp4"
# ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'temperature_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_temperature.mp4"
# ffmpeg -r 15 -f image2 -s 1920x1080 -pattern_type glob -i 'pressure_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_pressure.mp4"
ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'litho_new_temper_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_litho_new_temper.mp4"
ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'sp_surface_global_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_sp_surface_global.mp4"
ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'temper_mean_rm_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_temper_mean_rm.mp4"
ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i 'vs_anomaly_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_vs_anomaly.mp4"

ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i $DIRNAME'_lithology_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME'_lithology.mp4'
ffmpeg -r 15 -f image2 -s 1920x1080 -pattern_type glob -i 'Sim00*_lithology_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME'_lithology_15fps.mp4'


ffmpeg -r 15 -f image2 -s 1920x1080 -pattern_type glob -i $DIRNAME'_temperature_anomaly_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME'_temperature_anomaly_15fps.mp4'

ffmpeg -ss 0 -t 10 -i $DIRNAME'_temperature_anomaly.mp4' -vf "fps=15,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $DIRNAME'_temperature_anomaly_15fps.gif'

# ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i $DIRNAME'_strain_rate*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_strain_rate.mp4"

# rm viscosity_*.png heat_*.png density_*.png  strain_[0-9]*.png strain_rate*.png temperature_*.png litho_new_temper_*.png sp_surface_global_*.png pressure_*.png

# ffmpeg -r 15 -f image2 -s 1920x1080 -pattern_type glob -i 'sp_surface_global_zoom_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_sp_surface_global_zoom.mp4"

# ffmpeg -r 20 -f image2 -s 1920x1080 -i 'viscosity_%05d.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p $DIRNAME"_viscosity_tst.mp4"