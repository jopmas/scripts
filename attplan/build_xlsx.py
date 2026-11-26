#!/usr/bin/env python
'''
Cópia .xlsx da planilha de monitoramento a partir da exportação do attplan (JSON depois de @@EXPORT@@).

Uso: python build_xlsx.py saida_do_claude.txt destino.xlsx
'''
import json, os, sys, tempfile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

MARK = '@@EXPORT@@'
DEFAULT_STATUS = ['Planejado', 'Inputs prontos', 'Na fila', 'Rodando', 'Rodado', 'Pós-processado', 'Falhou']
CENTER = {'Status', 'Job ID (hypatia)', 'Regime', 'Velocidade (cm/yr)', 'Borda móvel', 'Tempo total (Myr)',
          'Tempo atingido (Myr)', 'Lx × Lz (km)', 'dx × dz (km)', 'Interfaces', 'Magmatism', 'Extraction', 'Ts0 (°C)',
          'Surface processes', 'TP (°C)', 'Quiescence (Myr)', 'RhoAst', 'RhoML_lc', 'CML_lc', 'RhoML_uc', 'CML_uc',
          'RhoML', 'CML', 'H_ML (W/kg)', 'RhoLC', 'High kappa', 'Rheol', 'Erosion', 'Grupo'}
WRAP = {'Comments', 'Comparison', 'Temperatura inicial'}
WIDTH = {'Cenário': 32, 'Status': 16, 'Job ID (hypatia)': 14, 'Grupo': 22, 'Pasta': 50, 'Comments': 45,
         'Comparison': 45, 'Temperatura inicial': 30, 'Surface processes': 18}


def plain(value):
    if isinstance(value, dict):
        for key in ('number', 'value', 'markdown', 'name'):
            if key in value:
                return plain(value[key])
        return None
    if isinstance(value, list):
        return ', '.join(str(plain(v)) for v in value)
    return value


def read_export(path):
    text = open(path, errors='replace').read()
    if MARK not in text:
        raise ValueError(f'{MARK} não está na saída do Claude')
    payload = text.split(MARK, 1)[1].strip()
    if payload.startswith('```'):
        payload = payload.strip('`').split('\n', 1)[1]
    start, end = payload.find('{'), payload.rfind('}')
    return json.loads(payload[start:end + 1])


def build(data, destination):
    wb = Workbook()
    wb.remove(wb.active)
    for sheet in data['sheets']:
        ws = wb.create_sheet(sheet['name'][:31])
        rows = sheet.get('rows') or []
        for row in rows:
            ws.append([plain(v) for v in row])
        if not rows:
            continue
        header = [str(h) if h is not None else '' for h in rows[0]]
        col = {name: i + 1 for i, name in enumerate(header)}
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill('solid', fgColor='F3F3F3')
            cell.alignment = Alignment(wrap_text=True, horizontal='center', vertical='center')
        for r in range(2, ws.max_row + 1):
            ws.cell(r, 1).font = Font(bold=True)
            for name, c in col.items():
                if name in CENTER:
                    ws.cell(r, c).alignment = Alignment(horizontal='center', vertical='top')
                elif name in WRAP:
                    ws.cell(r, c).alignment = Alignment(wrap_text=True, vertical='top')
            if 'H_ML (W/kg)' in col:
                ws.cell(r, col['H_ML (W/kg)']).number_format = '0.0E+00'
            # tempo atingido em vermelho: o run parou mais de 2 Myr antes do fim
            if 'Tempo atingido (Myr)' in col and 'Tempo total (Myr)' in col:
                reached = ws.cell(r, col['Tempo atingido (Myr)']).value
                total = ws.cell(r, col['Tempo total (Myr)']).value
                status = ws.cell(r, col['Status']).value if 'Status' in col else None
                if (isinstance(reached, (int, float)) and isinstance(total, (int, float)) and reached < total - 2
                        and status not in ('Rodando', 'Na fila')):
                    ws.cell(r, col['Tempo atingido (Myr)']).font = Font(color='C62828')
        if 'Status' in col:
            options = sheet.get('status_options') or DEFAULT_STATUS
            letter = get_column_letter(col['Status'])
            dv = DataValidation(type='list', formula1='"' + ','.join(options) + '"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f'{letter}2:{letter}500')
        for name, c in col.items():
            ws.column_dimensions[get_column_letter(c)].width = WIDTH.get(name, 12)
        ws.freeze_panes = 'B2'
    folder = os.path.dirname(destination)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix='.xlsx', dir=folder)  # troca atômica: o iCloud nunca vê um arquivo pela metade
    os.close(fd)
    wb.save(tmp)
    os.replace(tmp, destination)


if __name__ == '__main__':
    data = read_export(sys.argv[1])
    build(data, sys.argv[2])
    print(f"attplan: cópia salva em {sys.argv[2]} ({len(data['sheets'])} abas)")
