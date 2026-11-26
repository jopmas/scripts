#!/usr/bin/env bash
DIRNAME=${PWD##*/}
# Primeiro zipa os arquivos fixos
zip "$DIRNAME.zip" interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh
# Lista de padrões
patterns=(
"bc_velocity_*.txt"
"density"
"heat"
"pressure"
"surface"
"strain"
"temperature"
"time"
"velocity"
"viscosity"
"scale_bcv.txt"
"steps"
"lithos"
"Phi"
"dPhi"
"X_depletion"
"*.bin*.txt"
"bc*-1.txt"
)
# Faz um loop e usa find para evitar o erro "argument list too long"
for pat in "${patterns[@]}"; do
find . -maxdepth 1 -name "$pat" -exec zip -u "$DIRNAME.zip" {} +
done
