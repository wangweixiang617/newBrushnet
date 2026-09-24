#!/usr/bin/env python3
"""Cartesian Band[4] x Stage[5] x Time[10] routing sweep.

The scan config contains arrays of complete gain vectors. Every combination of
one band vector, one stage vector and one time vector is evaluated.
"""
import argparse
import csv
import itertools
import json
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
        if any(v < 0 for v in row):
            raise ValueError(f'{name}[{i}] contains a negative gain')
        out.append(row)
    return out


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


def main():
    parser = argparse.ArgumentParser(
        description='Run Cartesian combinations of Band[4], Stage[5], Time[10] gain arrays.'
    )
    parser.add_argument('--config', required=True, help='JSON containing band_sets/stage_sets/time_sets')
    parser.add_argument('--eval_script', default='evaluate_brushnet1_wave.py')
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--gpus', default='0', help='Comma-separated physical GPU IDs, e.g. 0,1,2,3')
    parser.add_argument('--rerun', action='store_true', help='Run even if evaluation_result_sum.csv exists')
    parser.add_argument('eval_args', nargs=argparse.REMAINDER,
                        help='Arguments passed to evaluator after --')
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    band_sets = _validate_sets('band_sets', config.get('band_sets'), 4)
    stage_sets = _validate_sets('stage_sets', config.get('stage_sets'), 5)
    time_sets = _validate_sets('time_sets', config.get('time_sets'), 10)

    gpus = [x.strip() for x in args.gpus.split(',') if x.strip()]
    if not gpus:
        raise ValueError('--gpus must contain at least one GPU ID')

    forwarded = list(args.eval_args)
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    forbidden = {'--image_save_path', '--wave_band_gains', '--wave_stage_gains', '--wave_time_gains'}
    if any(x in forbidden for x in forwarded):
        raise ValueError(
            'Do not pass --image_save_path or routing gain args after --; '
            'the scan runner supplies them per trial.'
        )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    trials = []
    for bi, si, ti in itertools.product(
        range(len(band_sets)), range(len(stage_sets)), range(len(time_sets))
    ):
        trial_id = f'b{bi:02d}_s{si:02d}_t{ti:02d}'
        trials.append({
            'trial_id': trial_id,
            'band_index': bi,
            'stage_index': si,
            'time_index': ti,
            'band_gains': band_sets[bi],
            'stage_gains': stage_sets[si],
            'time_gains': time_sets[ti],
            'output': str(out_root / trial_id),
        })

    manifest = {
        'config_path': str(Path(args.config).resolve()),
        'eval_script': str(Path(args.eval_script).resolve()),
        'gpus': gpus,
        'num_band_sets': len(band_sets),
        'num_stage_sets': len(stage_sets),
        'num_time_sets': len(time_sets),
        'num_trials': len(trials),
        'trials': trials,
    }
    (out_root / 'scan_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(
        f'[SCAN] {len(band_sets)} band x {len(stage_sets)} stage x '
        f'{len(time_sets)} time = {len(trials)} trials'
    )

    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def run_trial(trial):
        output = Path(trial['output'])
        summary = output / 'evaluation_result_sum.csv'
        if summary.exists() and not args.rerun:
            return trial['trial_id'], 0, 'SKIP', _read_summary(summary)

        output.mkdir(parents=True, exist_ok=True)
        gpu = gpu_queue.get()
        try:
            cmd = [
                sys.executable,
                args.eval_script,
                '--image_save_path', str(output),
                '--wave_band_gains', *_fmt(trial['band_gains']),
                '--wave_stage_gains', *_fmt(trial['stage_gains']),
                '--wave_time_gains', *_fmt(trial['time_gains']),
                *forwarded,
            ]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            # Make evaluator output visible immediately in the parent terminal.
            env['PYTHONUNBUFFERED'] = '1'

            print(
                f"\n[START] GPU={gpu} {trial['trial_id']} -> {output}\n"
                f"  Band  = {trial['band_gains']}\n"
                f"  Stage = {trial['stage_gains']}\n"
                f"  Time  = {trial['time_gains']}",
                flush=True,
            )

            # Do not redirect stdout/stderr to run.log.  The evaluator inherits
            # this terminal directly, so tqdm/progress bars, print() output and
            # tracebacks are shown live.  With multiple GPUs, outputs from
            # concurrent evaluators can interleave; this is expected.
            proc = subprocess.run(
                cmd,
                env=env,
            )

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

    results.sort(key=lambda r: r['trial_id'])
    (out_root / 'scan_results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    metric_names = sorted({k for r in results for k in r['metrics']})
    csv_path = out_root / 'scan_results.csv'
    fieldnames = [
        'trial_id', 'status', 'returncode', 'band_index', 'stage_index', 'time_index',
        'band_gains', 'stage_gains', 'time_gains', 'output', *metric_names
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                'trial_id': r['trial_id'],
                'status': r['status'],
                'returncode': r['returncode'],
                'band_index': r['band_index'],
                'stage_index': r['stage_index'],
                'time_index': r['time_index'],
                'band_gains': json.dumps(r['band_gains']),
                'stage_gains': json.dumps(r['stage_gains']),
                'time_gains': json.dumps(r['time_gains']),
                'output': r['output'],
            }
            row.update(r['metrics'])
            writer.writerow(row)

    failures = [r for r in results if r['returncode'] != 0]
    print(f'[DONE] results: {csv_path}')
    if failures:
        print(f'[WARN] {len(failures)} trial(s) failed; see the live terminal output above')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
