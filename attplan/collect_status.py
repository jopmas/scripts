#!/usr/bin/env python
'''
Estado dos cenários Mandyoc para a planilha de monitoramento (attplan).

Varre as pastas locais dos cenários (uma pasta com param.txt é um cenário) e os jobs do usuário no hypatia
(sacct, mais o último time_*.txt de cada pasta de run em /scratch) e escreve um JSON com, para cada cenário:
a aba da planilha, o status, o job ID, o tempo total e o tempo atingido, e os parâmetros das colunas da planilha
(para cenários que ainda não estão nela).

Uso: python collect_status.py [-o estado.json]
'''
import argparse, glob, json, os, re, subprocess, sys

ROOTS = ['/Volumes/Joao_Macedo/DB_tests_SVG', '/Volumes/Joao_Macedo/tests_rft', '/Volumes/Joao_Macedo/tests_svg',
         '/Volumes/Joao_Macedo/db_testes/CORRECTED']
CLASSIC_ROOT, CLASSIC_TAB = ROOTS[-1], 'db_tests_classic'  # cenários dos scripts generate_input_files_db_*.py
HYPATIA = 'jpmacedo@hypatia.iag.usp.br'
SACCT_START = '2026-09-01'
FAILED = ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')


def read_params(path):
    params = {}
    for line in open(path, errors='replace'):
        line = line.split('#')[0]
        if '=' in line:
            key, value = line.split('=', 1)
            params[key.strip()] = value.strip()
    return params


def classic_columns(scenario_dir, params, text):
    '''Colunas da aba db_tests_classic para os cenários dos scripts double_keel (11 camadas em interfaces.txt).'''
    layers = {}
    with open(os.path.join(scenario_dir, 'interfaces.txt')) as f:
        for _, line in zip(range(7), f):
            key, *values = line.split()
            layers[key] = [float(v) for v in values]
    quiescence = re.search(r'Time of quiescence: ([\d.]+)', text) or re.match(r'M(\d+)', os.path.basename(scenario_dir))
    tp = re.search(r'Potential Temperature for diffusive model: ([\d.]+)', text)
    columns = {
        'Quiescence (Myr)': float(quiescence.group(1)) if quiescence else None,
        'High kappa': 'Yes' if params.get('high_kappa_in_asthenosphere') == 'True' else 'No',
        'TP (°C)': float(tp.group(1)) if tp else None,
        'Rheol': int(params['rheology_model']) if 'rheology_model' in params else None,
        'Erosion': 'Yes' if params.get('sp_surface_processes') == 'True' else 'No',
        'Temperatura inicial': f"Difusão 1D (cp = {float(params['heat_capacity']):g})" if 'heat_capacity' in params else None,
    }
    if len(layers.get('rho', [])) == len(layers.get('C', [])) == len(layers.get('H', [])) == 11:
        rho, C, H = layers['rho'], layers['C'], layers['H']
        columns.update({'RhoAst': rho[0], 'RhoML_lc': rho[1], 'CML_lc': C[1], 'RhoML_uc': rho[2], 'CML_uc': C[2],
                        'RhoML': rho[3], 'CML': C[3], 'H_ML (W/kg)': H[3], 'RhoLC': rho[4]})
    return columns


def tab_of(root, scenario_dir):
    '''Aba da planilha: DB_tests_SVG e tests_rft têm uma aba cada; em tests_svg, uma aba por grupo (subpasta).'''
    if root == CLASSIC_ROOT:
        return CLASSIC_TAB
    base = os.path.basename(root)
    if base != 'tests_svg':
        return base
    group = os.path.relpath(scenario_dir, root).split(os.sep)[0]
    return group[:-len('_scenarios')] if group.endswith('_scenarios') else group


def local_scenario(root, scenario_dir):
    name = os.path.basename(scenario_dir)
    files = os.listdir(scenario_dir)
    params = read_params(os.path.join(scenario_dir, 'param.txt'))
    infos = [f for f in files if f.startswith('infos_') and 'track' not in f and f.endswith('.txt')]
    text = open(os.path.join(scenario_dir, infos[0]), errors='replace').read() if infos else ''

    def find(pattern):
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None

    output = os.path.join(scenario_dir, '_output')
    outputs = os.listdir(output) if os.path.isdir(output) else []
    nc = [f for f in files if f.endswith('.nc')]
    if any(f.endswith(('gifs.zip', 'imgs.zip')) for f in outputs):
        status = 'Pós-processado'
    elif nc or any(f.startswith('slurm') for f in files):
        status = 'Rodado'
    else:
        status = 'Inputs prontos'

    reached = None
    if 'times.csv' in files:
        lines = open(os.path.join(scenario_dir, 'times.csv')).read().strip().splitlines()
        if len(lines) > 1:
            reached = float(lines[-1].split(',')[1])
    if reached is None:
        for f in ['temperature.nc', 'density.nc', 'viscosity.nc', 'pressure.nc']:
            if f in files:
                try:
                    import xarray as xr
                    with xr.open_dataset(os.path.join(scenario_dir, f)) as dataset:
                        reached = float(dataset.time.values[-1])
                    break
                except Exception:
                    pass

    nx, nz = int(params['nx']), int(params['nz'])
    lx, lz = float(params['lx']) / 1e3, float(params['lz']) / 1e3
    preset = find(r'Preliminary scenario: (.*)')
    velocity = find(r'initial velocity: ([\d.]+)')
    sp = params.get('sp_surface_processes', 'False') == 'True'
    jobs = sorted({int(m.group(1)) for f in files + outputs for m in [re.search(r'(?:slurm|log_error)[-_]?(\d+)', f)] if m})
    scenario = {
        'name': name, 'tab': tab_of(root, scenario_dir), 'group': os.path.relpath(scenario_dir, root).split(os.sep)[0],
        'folder': os.path.relpath(scenario_dir, os.path.dirname(root)),
        'local_status': status, 'local_jobs': jobs,
        'time_max_myr': float(params.get('time_max', 0)) / 1e6, 'time_reached_local_myr': reached,
        'columns': {
            'Regime': find(r'Numerical experiment: (\S+)'),
            'Velocidade (cm/yr)': float(velocity) if velocity else None,
            'Borda móvel': find(r'moving boundary: (\w+)'),
            'Lx × Lz (km)': f'{lx:g} × {lz:g}', 'dx × dz (km)': f'{lx/(nx-1):g} × {lz/(nz-1):g}',
            'Interfaces': int(params['n_interfaces']) if 'n_interfaces' in params else None,
            'Magmatism': params.get('magmatism', 'off'), 'Extraction': params.get('magmatism_extraction'),
            'Ts0 (°C)': float(params['magmatism_Ts0']) if 'magmatism_Ts0' in params else None,
            'Surface processes': (f"Yes ({params['sp_mode']})" if sp and params.get('sp_mode') else 'Yes') if sp else 'No',
            'TP (°C)': float(find(r'Mantle potential temperature.*?: ([\d.]+)')) if find(r'Mantle potential temperature.*?: ([\d.]+)') else None,
            'Temperatura inicial': (f"Campo de {preset.split(' (')[0]}" if preset else ('Columns (tapioca)' if infos else None)),
        },
    }
    if root == CLASSIC_ROOT:
        scenario['folder'] = os.path.relpath(scenario_dir, os.path.dirname(os.path.dirname(root)))
        scenario['columns'] = classic_columns(scenario_dir, params, text)
        # pasta old/: cenários arquivados, atualizados só se já estiverem na aba
        scenario['update_only'] = scenario['group'] == 'old'
    return scenario


def hypatia_jobs():
    '''Jobs do usuário no hypatia (sacct) e o último tempo [Myr] de cada pasta de run.'''
    remote = (f"sacct -u $USER -S {SACCT_START} -X -n -P -o JobID,JobName,State,Submit,End,WorkDir; echo '@@TIMES'; "
              "for d in $(sacct -u $USER -S " + SACCT_START + " -X -n -P -o WorkDir | sort -u); do "
              # time_*.txt na pasta do run ou em time/ (para onde 0_organize_outputs.sh os move no fim do run): o maior passo
              "f=$(cd $d 2>/dev/null && ls . time 2>/dev/null | grep '^time_[0-9]*\\.txt$' | sort -t_ -k2 -n | tail -1); "
              # depois do pós-processamento só resta o times.csv (Myr na 2a coluna; aqui em anos, como nos time_*.txt)
              "if [ -z \"$f\" ]; then [ -f $d/times.csv ] && t=$(tail -n 1 $d/times.csv | cut -d, -f2) && "
              "echo \"$d|[0]  0:   ${t}e6\"; continue; fi; p=$d/$f; [ -f $p ] || p=$d/time/$f; "
              "echo \"$d|$(head -c 40 $p | tr '\\n' ' ')\"; done")
    try:
        out = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HYPATIA, remote],
                             capture_output=True, text=True, timeout=180)
    except Exception as err:
        return None, f'ssh falhou: {err}'
    if out.returncode != 0 and '@@TIMES' not in out.stdout:
        return None, f'ssh falhou: {out.stderr.strip()[:300]}'
    jobs_part, _, times_part = out.stdout.partition('@@TIMES')
    jobs = []
    for line in jobs_part.strip().splitlines():
        fields = line.split('|')
        if len(fields) >= 6 and fields[0].isdigit():
            jobs.append({'job': int(fields[0]), 'name': fields[1], 'state': fields[2].split()[0], 'submit': fields[3],
                         'end': fields[4], 'workdir': fields[5]})
    times = {}
    for line in times_part.strip().splitlines():
        workdir, _, head = line.partition('|')
        match = re.search(r'0:\s*([0-9.eE+-]+)', head)
        if match:
            times[workdir] = float(match.group(1)) / 1e6
    return {'jobs': jobs, 'times': times}, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output', help='arquivo JSON (padrão: saída padrão)')
    args = parser.parse_args(argv)

    scenarios = []
    for root in ROOTS:
        if not os.path.isdir(root):
            print(f'Aviso: {root} não está acessível (HD desconectado?).', file=sys.stderr)
            continue
        for param in sorted(glob.glob(os.path.join(root, '**', 'param.txt'), recursive=True)):
            if '/_output/' in param or '/inputs_' in param:
                continue
            scenarios.append(local_scenario(root, os.path.dirname(param)))

    hypatia, error = hypatia_jobs()
    for s in scenarios:
        status, job, reached, state = s['local_status'], None, s['time_reached_local_myr'], None
        if s['local_jobs']:
            job = s['local_jobs'][-1]
        if hypatia:
            mine = [j for j in hypatia['jobs'] if os.path.basename(j['workdir'].rstrip('/')) == s['name']]
            if mine:
                last = max(mine, key=lambda j: j['job'])
                job, state = last['job'], last['state']
                remote_time = hypatia['times'].get(last['workdir'])
                if state == 'RUNNING':
                    status = 'Rodando'
                    reached = remote_time if remote_time is not None else reached
                elif state == 'PENDING':
                    status, reached = 'Na fila', None
                elif state.startswith(FAILED) and status != 'Pós-processado':
                    status = 'Falhou'
                elif state == 'COMPLETED' and status == 'Inputs prontos':
                    status = 'Rodado'
                if state != 'RUNNING' and remote_time is not None and (reached is None or remote_time > reached):
                    reached = remote_time
        s.update({'status': status, 'job_id': job, 'hypatia_state': state,
                  'time_reached_myr': round(reached, 3) if reached is not None else None})

    result = {'hypatia': 'ok' if hypatia else error, 'scenarios': scenarios}
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.output:
        open(args.output, 'w').write(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
