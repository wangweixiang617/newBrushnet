#!/usr/bin/env python3
"""Build fixed per-slot/per-timestep global C tables from early-good trace JSON files.

For each input trace independently:
  1. bin calls by diffusion timestep,
  2. compute Q_p of wave_rms for every slot/bin.
Across traces, take the maximum Q_p, then multiply by the per-slot safety margin.
This implements C[i,b] = alpha[i] * max_k Q_p^{(k)}[i,b].
"""
import argparse
import json
from pathlib import Path

import torch

from wavebrush.residual_envelope import DEFAULT_SLOT_MARGINS


def timestep_bin(t, num_train_timesteps, num_bins):
    t = max(0, min(num_train_timesteps - 1, int(round(float(t)))))
    reverse = (num_train_timesteps - 1) - t
    return min(num_bins - 1, (reverse * num_bins) // num_train_timesteps)


def trace_quantiles(path, num_slots, num_bins, timesteps, quantile):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Trace {path} must contain a non-empty JSON list")
    values = [[[] for _ in range(num_bins)] for _ in range(num_slots)]
    for row in rows:
        b = timestep_bin(row["t"], timesteps, num_bins)
        residuals = row.get("residuals", [])
        if len(residuals) != num_slots:
            raise ValueError(f"Trace {path}: expected {num_slots} residual slots, got {len(residuals)}")
        for i, residual in enumerate(residuals):
            values[i][b].append(float(residual["wave_rms"]))

    out = []
    for i in range(num_slots):
        row = []
        for b in range(num_bins):
            if not values[i][b]:
                raise ValueError(f"Trace {path}: slot {i}, bin {b} has no samples")
            x = torch.tensor(values[i][b], dtype=torch.float32)
            row.append(float(torch.quantile(x, quantile)))
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True,
                        help="Early-good trace JSON. Repeat for step1000, step2500, etc.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--slots", type=int, default=28)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--margins", type=float, nargs=28, default=None,
                        help="Optional 28 alpha_i values; defaults to recommended margins")
    args = parser.parse_args()

    if not 0 < args.quantile <= 1:
        raise ValueError("--quantile must be in (0,1]")
    margins = list(args.margins) if args.margins is not None else list(DEFAULT_SLOT_MARGINS)
    if len(margins) != args.slots:
        raise ValueError(f"Expected {args.slots} margins, got {len(margins)}")

    per_trace = [
        trace_quantiles(p, args.slots, args.bins, args.timesteps, args.quantile)
        for p in args.trace
    ]
    qref = []
    caps = []
    for i in range(args.slots):
        qrow, crow = [], []
        for b in range(args.bins):
            q = max(table[i][b] for table in per_trace)
            qrow.append(q)
            crow.append(float(q * margins[i]))
        qref.append(qrow)
        caps.append(crow)

    data = {
        "num_slots": args.slots,
        "num_bins": args.bins,
        "num_train_timesteps": args.timesteps,
        "metadata": {
            "source_traces": [str(Path(p)) for p in args.trace],
            "quantile": args.quantile,
            "rule": "C[i,b] = alpha[i] * max_trace(P_quantile(wave_rms[i,b]))",
            "slot_margins": margins,
            "qref_global": qref,
        },
        "caps": {"global": caps},
    }
    Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
