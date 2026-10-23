#!/usr/bin/env bash
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
# Para cada padrão, procurar e remover com segurança
for pat in "${patterns[@]}"; do
find . -maxdepth 1 -name "$pat" -exec rm -f {} +
done
