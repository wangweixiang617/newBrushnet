#!/usr/bin/env python3
"""Build global/known/hole per-slot/per-timestep residual-envelope cap tables.

Input traces must be produced by the patched wavebrush.integration below.  Each
residual slot contains sample-level RMS values for global, known and hole
regions.  For every trace independently we compute Q_p for each slot/bin;
across early-good traces we take the maximum Q_p and multiply by alpha:

    C_r[i,b] = alpha_r * max_k Q_p^(k)(r, i, b)

where r is global / known / hole.
"""
import argparse
import json
from pathlib import Path

import torch

REGIONS = ("global", "known", "hole")
SAMPLE_KEYS = {
    "global": "wave_rms_samples",
    "known": "known_rms_samples",
    "hole": "hole_rms_samples",
}


def timestep_bin(t, num_train_timesteps, num_bins):
    t = max(0, min(num_train_timesteps - 1, int(round(float(t)))))
    reverse = (num_train_timesteps - 1) - t
    return min(num_bins - 1, (reverse * num_bins) // num_train_timesteps)


def _values_for_region(residual, region, path):
    key = SAMPLE_KEYS[region]
    values = residual.get(key)
    if values is None:
        if region == "global" and "wave_rms" in residual:
            # Backward-compatible global-only fallback for old traces.
            return [float(residual["wave_rms"])]
        raise ValueError(
            f"Trace {path} has no '{key}'. Known/hole caps require traces generated "
            "with the region-RMS trace patch; regenerate the early-good validation trace."
        )
    if not isinstance(values, list):
        raise ValueError(f"Trace {path}: '{key}' must be a list")
    return [float(v) for v in values if v is not None]


def trace_quantiles(path, regions, num_slots, num_bins, timesteps, quantile):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Trace {path} must contain a non-empty JSON list")

    values = {
        region: [[[] for _ in range(num_bins)] for _ in range(num_slots)]
        for region in regions
    }
    min_region_fractions = set()

    for row in rows:
        b = timestep_bin(row["t"], timesteps, num_bins)
        if "region_min_fraction" in row:
            min_region_fractions.add(float(row["region_min_fraction"]))
        residuals = row.get("residuals", [])
        if len(residuals) != num_slots:
            raise ValueError(
                f"Trace {path}: expected {num_slots} residual slots, got {len(residuals)}"
            )
        for i, residual in enumerate(residuals):
            for region in regions:
                values[region][i][b].extend(_values_for_region(residual, region, path))

    tables = {}
    counts = {}
    for region in regions:
        qtable, ctable = [], []
        for i in range(num_slots):
            qrow, crow = [], []
            for b in range(num_bins):
                cell = values[region][i][b]
                if not cell:
                    raise ValueError(
                        f"Trace {path}: region={region}, slot={i}, bin={b} has no valid samples. "
                        "Use more reference batches or reduce --wave_env_min_region_fraction when collecting."
                    )
                x = torch.tensor(cell, dtype=torch.float32)
                qrow.append(float(torch.quantile(x, quantile)))
                crow.append(len(cell))
            qtable.append(qrow)
            ctable.append(crow)
        tables[region] = qtable
        counts[region] = ctable

    return tables, counts, sorted(min_region_fractions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", action="append", required=True,
        help="Early-good patched validation trace JSON. Repeat for step1000, step2000, etc."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--slots", type=int, default=28)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--alpha-global", type=float, default=1.1)
    parser.add_argument("--alpha-known", type=float, default=1.1)
    parser.add_argument("--alpha-hole", type=float, default=1.1)
    args = parser.parse_args()

    if not 0 < args.quantile <= 1:
        raise ValueError("--quantile must be in (0,1]")
    if args.slots < 1 or args.bins < 1 or args.timesteps < 1:
        raise ValueError("--slots, --bins and --timesteps must be positive")

    regions = list(dict.fromkeys(args.regions))
    alphas = {
        "global": float(args.alpha_global),
        "known": float(args.alpha_known),
        "hole": float(args.alpha_hole),
    }
    for region in regions:
        if not torch.isfinite(torch.tensor(alphas[region])) or alphas[region] <= 0:
            raise ValueError(f"alpha for {region} must be finite and > 0")

    per_trace = []
    per_trace_counts = []
    trace_min_fracs = []
    for path in args.trace:
        tables, counts, min_fracs = trace_quantiles(
            path, regions, args.slots, args.bins, args.timesteps, args.quantile
        )
        per_trace.append(tables)
        per_trace_counts.append(counts)
        trace_min_fracs.append(min_fracs)

    qref = {}
    caps = {}
    sample_counts = {}
    for region in regions:
        qtable, ctable, ntable = [], [], []
        for i in range(args.slots):
            qrow, crow, nrow = [], [], []
            for b in range(args.bins):
                trace_qs = [table[region][i][b] for table in per_trace]
                q = max(trace_qs)
                qrow.append(q)
                crow.append(float(q * alphas[region]))
                nrow.append([counts[region][i][b] for counts in per_trace_counts])
            qtable.append(qrow)
            ctable.append(crow)
            ntable.append(nrow)
        qref[region] = qtable
        caps[region] = ctable
        sample_counts[region] = ntable

    data = {
        "num_slots": args.slots,
        "num_bins": args.bins,
        "num_train_timesteps": args.timesteps,
        "metadata": {
            "source_traces": [str(Path(p)) for p in args.trace],
            "regions": regions,
            "quantile": args.quantile,
            "rule": "C_r[i,b] = alpha_r * max_trace(P_quantile(sample_RMS_r[i,b]))",
            "alphas": {r: alphas[r] for r in regions},
            "qref": qref,
            "sample_counts_per_trace": sample_counts,
            "trace_region_min_fraction": trace_min_fracs,
        },
        "caps": caps,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(args.output)
    for region in regions:
        flat = torch.tensor(caps[region], dtype=torch.float32)
        print(
            f"{region}: shape={tuple(flat.shape)} alpha={alphas[region]:.4f} "
            f"min={float(flat.min()):.6f} mean={float(flat.mean()):.6f} max={float(flat.max()):.6f}"
        )


if __name__ == "__main__":
    main()
