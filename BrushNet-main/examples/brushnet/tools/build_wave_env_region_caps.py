#!/usr/bin/env python3
"""Build topology-aware WaveBrush residual-envelope cap tables.

Input traces contain the full public BrushNet residual schema (normally 28
slots), but only active Wave injection slots are calibrated. Inactive slots are
written as 0.0 placeholders and never contribute to the envelope loss.

For every active slot / timestep bin and trace independently:

    C_r[i,b] = alpha_r * max_k Q_p^(k)(sample_RMS_r[i,b])

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


def _normalize_active_slots(values, num_slots, label):
    try:
        values = [int(i) for i in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: active_slot_indices must contain integers") from exc
    if not values:
        raise ValueError(f"{label}: active_slot_indices must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{label}: active_slot_indices contains duplicates: {values}")
    bad = [i for i in values if i < 0 or i >= num_slots]
    if bad:
        raise ValueError(f"{label}: active_slot_indices contains out-of-range slots: {bad}")
    return values


def _trace_topology(rows, path, num_slots):
    explicit = []
    inferred = []
    modes = set()
    signatures = set()

    for row_index, row in enumerate(rows):
        residuals = row.get("residuals", [])
        if len(residuals) != num_slots:
            raise ValueError(
                f"Trace {path}: expected {num_slots} residual slots, got {len(residuals)} at row {row_index}"
            )

        if "active_slot_indices" in row:
            active = _normalize_active_slots(
                row["active_slot_indices"], num_slots, f"Trace {path} row {row_index}"
            )
            explicit.append(active)

        flags = [r.get("wave_active") for r in residuals]
        if any(flag is not None for flag in flags):
            if not all(flag is not None for flag in flags):
                raise ValueError(f"Trace {path} row {row_index}: wave_active must be present for every residual row")
            active = [i for i, flag in enumerate(flags) if bool(flag)]
            inferred.append(_normalize_active_slots(active, num_slots, f"Trace {path} row {row_index}"))

        if row.get("inject_mode") is not None:
            modes.add(str(row["inject_mode"]))
        if row.get("topology_signature") is not None:
            signatures.add(str(row["topology_signature"]))

    candidates = explicit + inferred
    if candidates:
        active = candidates[0]
        for other in candidates[1:]:
            if other != active:
                raise ValueError(
                    f"Trace {path}: inconsistent active-slot topology inside one trace: {active} vs {other}"
                )
    else:
        # Backward compatibility for pre-topology traces: all public slots active.
        active = list(range(num_slots))

    if len(modes) > 1:
        raise ValueError(f"Trace {path}: multiple inject_mode values found: {sorted(modes)}")
    if len(signatures) > 1:
        raise ValueError(f"Trace {path}: multiple topology_signature values found: {sorted(signatures)}")

    return {
        "active_slot_indices": active,
        "inactive_slot_indices": [i for i in range(num_slots) if i not in set(active)],
        "inject_mode": next(iter(modes)) if modes else None,
        "topology_signature": next(iter(signatures)) if signatures else None,
    }


def trace_quantiles(path, regions, num_slots, num_bins, timesteps, quantile):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Trace {path} must contain a non-empty JSON list")

    topology = _trace_topology(rows, path, num_slots)
    active_set = set(topology["active_slot_indices"])
    values = {
        region: [[[] for _ in range(num_bins)] for _ in range(num_slots)]
        for region in regions
    }
    min_region_fractions = set()

    for row in rows:
        b = timestep_bin(row["t"], timesteps, num_bins)
        if "region_min_fraction" in row:
            min_region_fractions.add(float(row["region_min_fraction"]))
        residuals = row["residuals"]
        for i, residual in enumerate(residuals):
            if i not in active_set:
                continue
            for region in regions:
                values[region][i][b].extend(_values_for_region(residual, region, path))

    tables = {}
    counts = {}
    for region in regions:
        qtable, ctable = [], []
        for i in range(num_slots):
            qrow, crow = [], []
            for b in range(num_bins):
                if i not in active_set:
                    qrow.append(0.0)
                    crow.append(0)
                    continue
                cell = values[region][i][b]
                if not cell:
                    raise ValueError(
                        f"Trace {path}: region={region}, active slot={i}, bin={b} has no valid samples. "
                        "Use more reference batches or reduce --wave_env_min_region_fraction when collecting."
                    )
                x = torch.tensor(cell, dtype=torch.float32)
                qrow.append(float(torch.quantile(x, quantile)))
                crow.append(len(cell))
            qtable.append(qrow)
            ctable.append(crow)
        tables[region] = qtable
        counts[region] = ctable

    return tables, counts, sorted(min_region_fractions), topology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", action="append", required=True,
        help="Early-good validation trace JSON. Repeat for step1000, step2000, etc."
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
    topologies = []
    for path in args.trace:
        tables, counts, min_fracs, topology = trace_quantiles(
            path, regions, args.slots, args.bins, args.timesteps, args.quantile
        )
        per_trace.append(tables)
        per_trace_counts.append(counts)
        trace_min_fracs.append(min_fracs)
        topologies.append(topology)

    active = topologies[0]["active_slot_indices"]
    for path, topology in zip(args.trace[1:], topologies[1:]):
        if topology["active_slot_indices"] != active:
            raise ValueError(
                f"Trace topology mismatch: first trace active slots {active}, but {path} has "
                f"{topology['active_slot_indices']}. Build caps only from matching injection topologies."
            )
    active_set = set(active)
    inactive = [i for i in range(args.slots) if i not in active_set]

    qref = {}
    caps = {}
    sample_counts = {}
    for region in regions:
        qtable, ctable, ntable = [], [], []
        for i in range(args.slots):
            qrow, crow, nrow = [], [], []
            for b in range(args.bins):
                if i not in active_set:
                    qrow.append(0.0)
                    crow.append(0.0)
                    nrow.append([0 for _ in per_trace_counts])
                    continue
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

    inject_modes = [t["inject_mode"] for t in topologies]
    topology_signatures = [t["topology_signature"] for t in topologies]
    data = {
        "num_slots": args.slots,
        "num_bins": args.bins,
        "num_train_timesteps": args.timesteps,
        "metadata": {
            "source_traces": [str(Path(p)) for p in args.trace],
            "regions": regions,
            "quantile": args.quantile,
            "rule": "C_r[i,b] = alpha_r * max_trace(P_quantile(sample_RMS_r[i,b])) for active slots only",
            "alphas": {r: alphas[r] for r in regions},
            "active_slot_indices": active,
            "inactive_slot_indices": inactive,
            "inject_modes_per_trace": inject_modes,
            "topology_signatures_per_trace": topology_signatures,
            "qref": qref,
            "sample_counts_per_trace": sample_counts,
            "trace_region_min_fraction": trace_min_fracs,
        },
        "caps": caps,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(args.output)
    print(f"active slots ({len(active)}/{args.slots}): {active}")
    for region in regions:
        active_values = torch.tensor(
            [caps[region][i] for i in active], dtype=torch.float32
        )
        print(
            f"{region}: active_shape={tuple(active_values.shape)} alpha={alphas[region]:.4f} "
            f"min={float(active_values.min()):.6f} mean={float(active_values.mean()):.6f} "
            f"max={float(active_values.max()):.6f}"
        )


if __name__ == "__main__":
    main()
