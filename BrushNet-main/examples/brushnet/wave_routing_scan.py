#!/usr/bin/env python3
"""Cartesian Wave routing sweep.

Supports separable Band[4], Stage[5], Time[10] routing plus optional
Band x Time [4x10], Band x Stage [4x5], and wave_strength sets.

The evaluator is launched once per Cartesian combination. Runtime routing is
inference-only; no model weights are trained by this runner.
"""
import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def _validate_sets(name, sets, width):
    if not isinstance(sets, list) or not sets:
        raise ValueError(f'{name} must be a non-empty list of arrays')
    out = []
    for i, values in enumerate(sets):
        if not isinstance(values, list) or len(values) != width:
            raise ValueError(f'{name}[{i}] must contain exactly {width} values')
        row = [float(v) for v in values]
        if any((not math.isfinite(v)) or v < 0 for v in row):
            raise ValueError(f'{name}[{i}] contains a non-finite or negative gain')
        out.append(row)
    return out


def _validate_matrix_sets(name, sets, rows, cols):
    """Accept a list of nested [rows][cols] matrices or flat rows*cols arrays."""
    if sets is None:
        return [[[1.0] * cols for _ in range(rows)]]
    if not isinstance(sets, list) or not sets:
        raise ValueError(f'{name} must be a non-empty list of matrices')

    out = []
    for i, values in enumerate(sets):
        if not isinstance(values, list):
            raise ValueError(f'{name}[{i}] must be a matrix or flat array')
        if len(values) == rows and all(isinstance(row, list) for row in values):
            matrix = []
            for r, row in enumerate(values):
                if len(row) != cols:
                    raise ValueError(f'{name}[{i}][{r}] must contain exactly {cols} values')
                matrix.append([float(v) for v in row])
        else:
            if len(values) != rows * cols:
                raise ValueError(
                    f'{name}[{i}] must have shape [{rows}, {cols}] or contain {rows * cols} flat values'
                )
            flat = [float(v) for v in values]
            matrix = [flat[r * cols:(r + 1) * cols] for r in range(rows)]
        if any((not math.isfinite(v)) or v < 0 for row in matrix for v in row):
            raise ValueError(f'{name}[{i}] contains a non-finite or negative gain')
        out.append(matrix)
    return out


def _validate_strengths(values):
    if values is None:
        return None
    if not isinstance(values, list) or not values:
        raise ValueError('wave_strengths must be a non-empty list')
    out = [float(v) for v in values]
    if any((not math.isfinite(v)) or v < 0 for v in out):
        raise ValueError('wave_strengths values must be finite and >= 0')
    return out


def _flatten(matrix):
    return [v for row in matrix for v in row]


def _read_summary(path):
    """Read pandas Series.to_csv() output: metric,value rows."""
    if not path.exists():
        return {}
    result = {}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0]:
                continue
            try:
                result[row[0]] = float(row[1])
            except ValueError:
                pass
    return result


def _fmt(values):
    return [format(float(v), '.8g') for v in values]


def _strength_tag(value):
    text = format(float(value), '.8g')
    return text.replace('-', 'm').replace('.', 'p').replace('+', '')


def _contains_option(args, option):
    return any(token == option or token.startswith(option + '=') for token in args)


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Run Cartesian combinations of Band[4], Stage[5], Time[10], '
            'BandTime[4x10], BandStage[4x5], and wave_strength arrays.'
        )
    )
    parser.add_argument('--config', required=True,
                        help='JSON containing routing sets and optional wave_strengths')
    parser.add_argument('--eval_script', default='evaluate_brushnet1_wave.py')
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--gpus', default='0', help='Comma-separated physical GPU IDs, e.g. 0,1,2,3')
    parser.add_argument('--rerun', action='store_true', help='Run even if evaluation_result_sum.csv exists')
    parser.add_argument('eval_args', nargs=argparse.REMAINDER,
                        help='Arguments passed to evaluator after --')
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    band_sets = _validate_sets('band_sets', config.get('band_sets'), 4)
    stage_sets = _validate_sets('stage_sets', config.get('stage_sets'), 5)
    time_sets = _validate_sets('time_sets', config.get('time_sets'), 10)
    band_time_sets = _validate_matrix_sets('band_time_sets', config.get('band_time_sets'), 4, 10)
    band_stage_sets = _validate_matrix_sets('band_stage_sets', config.get('band_stage_sets'), 4, 5)
    wave_strengths = _validate_strengths(config.get('wave_strengths'))
    include_baseline = config.get('include_baseline', False)
    if not isinstance(include_baseline, bool):
        raise ValueError('include_baseline must be true or false')

    gpus = [x.strip() for x in args.gpus.split(',') if x.strip()]
    if not gpus:
        raise ValueError('--gpus must contain at least one GPU ID')

    eval_script = Path(args.eval_script)
    if not eval_script.is_absolute():
        eval_script = eval_script.resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f'Evaluator script not found: {eval_script}')

    forwarded = list(args.eval_args)
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]

    runner_owned = {
        '--image_save_path', '--wave_band_gains', '--wave_stage_gains', '--wave_time_gains',
        '--wave_band_time_gains', '--wave_band_stage_gains', '--disable_wave',
    }
    if any(_contains_option(forwarded, option) for option in runner_owned):
        raise ValueError(
            'Do not pass image_save_path, routing gain args, or --disable_wave after --; '
            'the scan runner supplies them per trial.'
        )
    if wave_strengths is not None and _contains_option(forwarded, '--wave_strength'):
        raise ValueError(
            'wave_strengths is present in the JSON config, so do not also pass --wave_strength after --.'
        )

    # If wave_strengths is absent, preserve the old behavior: the runner does not
    # inject --wave_strength and the evaluator receives the forwarded/default value.
    strength_items = list(enumerate(wave_strengths)) if wave_strengths is not None else [(None, None)]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    trials = []
    for bi, si, ti, bti, bsi, strength_item in itertools.product(
        range(len(band_sets)),
        range(len(stage_sets)),
        range(len(time_sets)),
        range(len(band_time_sets)),
        range(len(band_stage_sets)),
        strength_items,
    ):
        ai, strength = strength_item
        trial_id = f'b{bi:02d}_s{si:02d}_t{ti:02d}_bt{bti:02d}_bs{bsi:02d}'
        if strength is not None:
            trial_id += f'_a{ai:02d}_{_strength_tag(strength)}'
        trials.append({
            'trial_id': trial_id,
            'trial_type': 'wave',
            'is_baseline': False,
            'band_index': bi,
            'stage_index': si,
            'time_index': ti,
            'band_time_index': bti,
            'band_stage_index': bsi,
            'wave_strength_index': ai,
            'wave_strength': strength,
            'band_gains': band_sets[bi],
            'stage_gains': stage_sets[si],
            'time_gains': time_sets[ti],
            'band_time_gains': band_time_sets[bti],
            'band_stage_gains': band_stage_sets[bsi],
            'output': str(out_root / trial_id),
        })

    num_wave_trials = len(trials)
    if include_baseline:
        baseline_id = 'baseline_brushnet'
        trials.append({
            'trial_id': baseline_id,
            'trial_type': 'baseline',
            'is_baseline': True,
            'band_index': None,
            'stage_index': None,
            'time_index': None,
            'band_time_index': None,
            'band_stage_index': None,
            'wave_strength_index': None,
            'wave_strength': None,
            'band_gains': None,
            'stage_gains': None,
            'time_gains': None,
            'band_time_gains': None,
            'band_stage_gains': None,
            'output': str(out_root / baseline_id),
        })

    manifest = {
        'config_path': str(config_path),
        'eval_script': str(eval_script),
        'gpus': gpus,
        'num_band_sets': len(band_sets),
        'num_stage_sets': len(stage_sets),
        'num_time_sets': len(time_sets),
        'num_band_time_sets': len(band_time_sets),
        'num_band_stage_sets': len(band_stage_sets),
        'wave_strengths': wave_strengths,
        'num_strengths': len(wave_strengths) if wave_strengths is not None else None,
        'include_baseline': include_baseline,
        'num_wave_trials': num_wave_trials,
        'num_baseline_trials': 1 if include_baseline else 0,
        'num_trials': len(trials),
        'trials': trials,
    }
    (out_root / 'scan_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    strength_count = len(wave_strengths) if wave_strengths is not None else 1
    strength_desc = f'{strength_count} strength' if wave_strengths is not None else 'CLI/default strength'
    baseline_desc = ' + 1 BrushNet baseline' if include_baseline else ''
    print(
        f'[SCAN] {len(band_sets)} band x {len(stage_sets)} stage x '
        f'{len(time_sets)} time x {len(band_time_sets)} band-time x '
        f'{len(band_stage_sets)} band-stage x {strength_desc} = {num_wave_trials} Wave trials'
        f'{baseline_desc}; total={len(trials)}',
        flush=True,
    )

    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def run_trial(trial):
        output = Path(trial['output'])
        summary = output / 'evaluation_result_sum.csv'
        if summary.exists() and not args.rerun:
            print(f"[SKIP] {trial['trial_id']} -> {output}", flush=True)
            return trial['trial_id'], 0, 'SKIP', _read_summary(summary)

        output.mkdir(parents=True, exist_ok=True)
        gpu = gpu_queue.get()
        try:
            if trial['is_baseline']:
                cmd = [
                    sys.executable,
                    str(eval_script),
                    '--image_save_path', str(output),
                    '--disable_wave',
                    *forwarded,
                ]
            else:
                cmd = [
                    sys.executable,
                    str(eval_script),
                    '--image_save_path', str(output),
                    '--wave_band_gains', *_fmt(trial['band_gains']),
                    '--wave_stage_gains', *_fmt(trial['stage_gains']),
                    '--wave_time_gains', *_fmt(trial['time_gains']),
                    '--wave_band_time_gains', *_fmt(_flatten(trial['band_time_gains'])),
                    '--wave_band_stage_gains', *_fmt(_flatten(trial['band_stage_gains'])),
                ]
                if trial['wave_strength'] is not None:
                    cmd += ['--wave_strength', format(float(trial['wave_strength']), '.8g')]
                cmd += forwarded

            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            env['PYTHONUNBUFFERED'] = '1'

            if trial['is_baseline']:
                print(
                    f"\n[START] GPU={gpu} {trial['trial_id']} -> {output}\n"
                    f"  Mode       = BrushNet baseline (--disable_wave)",
                    flush=True,
                )
            else:
                print(
                    f"\n[START] GPU={gpu} {trial['trial_id']} -> {output}\n"
                    f"  Strength   = {trial['wave_strength'] if trial['wave_strength'] is not None else 'CLI/default'}\n"
                    f"  Band       = {trial['band_gains']}\n"
                    f"  Stage      = {trial['stage_gains']}\n"
                    f"  Time       = {trial['time_gains']}\n"
                    f"  Band-Time  = {trial['band_time_gains']}\n"
                    f"  Band-Stage = {trial['band_stage_gains']}",
                    flush=True,
                )

            # Evaluator inherits this terminal so tqdm/print/tracebacks remain visible.
            proc = subprocess.run(cmd, env=env)

            status = 'OK' if proc.returncode == 0 else 'FAIL'
            print(
                f"\n[{status}] GPU={gpu} {trial['trial_id']} rc={proc.returncode}",
                flush=True,
            )
            return trial['trial_id'], proc.returncode, status, _read_summary(summary)
        finally:
            gpu_queue.put(gpu)

    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_trial, trial) for trial in trials]
        for future in as_completed(futures):
            trial_id, rc, status, metrics = future.result()
            trial = next(t for t in trials if t['trial_id'] == trial_id)
            results.append({**trial, 'returncode': rc, 'status': status, 'metrics': metrics})

    results.sort(key=lambda r: (0 if r['is_baseline'] else 1, r['trial_id']))
    (out_root / 'scan_results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    metric_names = sorted({k for r in results for k in r['metrics']})
    csv_path = out_root / 'scan_results.csv'
    fieldnames = [
        'trial_id', 'trial_type', 'is_baseline', 'status', 'returncode',
        'band_index', 'stage_index', 'time_index', 'band_time_index', 'band_stage_index',
        'wave_strength_index', 'wave_strength',
        'band_gains', 'stage_gains', 'time_gains', 'band_time_gains', 'band_stage_gains',
        'output', *metric_names,
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                'trial_id': r['trial_id'],
                'trial_type': r['trial_type'],
                'is_baseline': r['is_baseline'],
                'status': r['status'],
                'returncode': r['returncode'],
                'band_index': r['band_index'],
                'stage_index': r['stage_index'],
                'time_index': r['time_index'],
                'band_time_index': r['band_time_index'],
                'band_stage_index': r['band_stage_index'],
                'wave_strength_index': r['wave_strength_index'],
                'wave_strength': r['wave_strength'],
                'band_gains': '' if r['band_gains'] is None else json.dumps(r['band_gains']),
                'stage_gains': '' if r['stage_gains'] is None else json.dumps(r['stage_gains']),
                'time_gains': '' if r['time_gains'] is None else json.dumps(r['time_gains']),
                'band_time_gains': '' if r['band_time_gains'] is None else json.dumps(r['band_time_gains']),
                'band_stage_gains': '' if r['band_stage_gains'] is None else json.dumps(r['band_stage_gains']),
                'output': r['output'],
            }
            row.update(r['metrics'])
            writer.writerow(row)

    failures = [r for r in results if r['returncode'] != 0]
    print(f'[DONE] results: {csv_path}', flush=True)
    if failures:
        print(f'[WARN] {len(failures)} trial(s) failed; see the live terminal output above', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
