#!/bin/bash


# Se DIR não for passado, usa o diretório atual.

patterns=(
  "bc_velocity_*.txt"
  "density_*.txt"
  "heat_*.txt"
  "pressure_*.txt"
  "sp_surface_global_*.txt"
  "strain_*.txt"
  "temperature_*.txt"
  "time_*.txt"
  "velocity_*.txt"
  "viscosity_*.txt"
  "scale_bcv.txt"
  "step*.txt"
  "Phi*.txt"
  "dPhi*.txt"
  "X_depletion*.txt"
  "track_*.txt"
  "*.bin*.txt"
  "bc*-1.txt"
)

# Para cada padrão, procurar e remover com segurança
for pat in "${patterns[@]}"; do
  find . -maxdepth 1 -type f -name "$pat" -exec rm -f {} +
done
