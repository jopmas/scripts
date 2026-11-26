#!/usr/bin/env bash
# attplan: atualiza a planilha de monitoramento de cenários Mandyoc
# (Google Sheets Monitor_Scenarios, https://docs.google.com/spreadsheets/d/1yj0JtUXj8szrJiXnG5B_H0eIiiH2BoYmyOhlqxwIrTQ) com o estado das pastas locais e dos jobs no hypatia,
# e salva uma cópia .xlsx no iCloud Drive.
#
# Uso: attplan           coleta o estado, pede ao Claude Code que atualize a planilha e salva a cópia
#      attplan --dry-run só mostra o estado coletado (JSON), sem mexer na planilha
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/anaconda3/envs/tapioca/bin/python"
BASEPY="$HOME/anaconda3/bin/python"   # tem openpyxl
XLSX="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Monitor_cenarios_Mandyoc.xlsx"
STATE="$(mktemp -t attplan).json"
OUT="$(mktemp -t attplan).txt"
trap 'rm -f "$STATE" "$OUT"' EXIT

echo "attplan: coletando o estado das pastas e do hypatia..."
"$PY" -I "$DIR/collect_status.py" -o "$STATE"
N=$(grep -c '"name"' "$STATE" || true)
HYP=$(grep -m1 '"hypatia"' "$STATE" | sed -E 's/.*"hypatia": "?([^"]*)"?,?/\1/')
echo "attplan: $N cenários; hypatia: $HYP"

if [ "${1:-}" = "--dry-run" ]; then
    cat "$STATE"
    exit 0
fi

echo "attplan: atualizando a planilha (Claude Code)..."
cd "$DIR"
claude -p "$(cat "$DIR/prompt.md")" \
    --allowedTools "mcp__claude_ai_Google_Sheets__get_spreadsheet" "mcp__claude_ai_Google_Sheets__get_values" "mcp__claude_ai_Google_Sheets__update_values" "mcp__claude_ai_Google_Sheets__update_spreadsheet" \
    --output-format text < "$STATE" > "$OUT"

# o resumo vai para o terminal; a exportação (depois de @@EXPORT@@) vira o .xlsx do iCloud
sed '/@@EXPORT@@/,$d' "$OUT"
if "$BASEPY" -I "$DIR/build_xlsx.py" "$OUT" "$XLSX"; then
    :
else
    echo "attplan: AVISO: a planilha online foi atualizada, mas a cópia do iCloud não (exportação inválida)." >&2
    cp "$OUT" "$HOME/attplan_ultima_saida.txt"
    echo "attplan: saída do Claude guardada em ~/attplan_ultima_saida.txt" >&2
fi
