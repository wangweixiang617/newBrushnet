"""Per-slot residual-envelope regularization for WaveBrush.

The regularizer constrains Wave residual amplitude without changing the forward
architecture. It supports three independently weighted regions:
  * global: full residual feature map
  * known:  known/unmasked image region
  * hole:   masked/inpainting region

All public masks use 1=hole, matching wavebrush.core.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

REGIONS = ("global", "known", "hole")

# Recommended slot weights from the current 28-slot SD1.5 BrushNet analysis.
DEFAULT_SLOT_WEIGHTS = (
    0.75, 0.75, 0.75,
    1.00, 1.00, 1.00, 1.00,
    1.25, 1.25,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    2.00,
    1.00, 1.00, 1.00, 1.00,
    0.75, 0.75, 0.90,
    1.50, 1.50, 1.50,
    2.00, 2.00,
)

# Recommended margins used when converting an early-good reference Q95 into C.
# These are only used by the supplied cap-building utility; training consumes the
# final C values directly.
DEFAULT_SLOT_MARGINS = (
    1.15, 1.15, 1.15,
    1.10, 1.10, 1.10, 1.10,
    1.08, 1.08,
    1.10, 1.10, 1.10, 1.10, 1.10, 1.10,
    1.05,
    1.10, 1.10, 1.10, 1.10,
    1.15, 1.15, 1.12,
    1.05, 1.05, 1.05,
    1.03, 1.03,
)


def _as_slot_weights(values: Optional[Iterable[float]], num_slots: int, device) -> torch.Tensor:
    if values is None:
        if num_slots != len(DEFAULT_SLOT_WEIGHTS):
            raise ValueError(
                f"Default residual-envelope slot weights require {len(DEFAULT_SLOT_WEIGHTS)} slots; "
                f"current Wave has {num_slots}. Pass explicit per-slot weights."
            )
        values = DEFAULT_SLOT_WEIGHTS
    values = torch.as_tensor(list(values), dtype=torch.float32, device=device)
    if values.shape != (num_slots,):
        raise ValueError(f"Expected {num_slots} slot weights, got shape {tuple(values.shape)}")
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Residual-envelope slot weights must be finite and positive")
    return values


def _load_cap_file(path: Optional[str]) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Residual-envelope cap JSON must contain an object")
    return data


def _cap_table(
    region: str,
    fixed_c: Optional[float],
    cap_file: dict,
    num_slots: int,
    num_bins: int,
    num_train_timesteps: int,
    device,
    active_slot_indices=None,
) -> Optional[torch.Tensor]:
    # A region-specific fixed C intentionally overrides the JSON table for that region.
    if fixed_c is not None:
        fixed_c = float(fixed_c)
        if not torch.isfinite(torch.tensor(fixed_c)) or fixed_c <= 0:
            raise ValueError(f"--wave_env_{region}_c must be finite and > 0")
        return torch.full((num_slots, num_bins), fixed_c, dtype=torch.float32, device=device)

    if not cap_file:
        return None

    file_bins = int(cap_file.get("num_bins", num_bins))
    if file_bins != num_bins:
        raise ValueError(f"Cap JSON num_bins={file_bins} differs from --wave_env_bins={num_bins}")
    file_slots = int(cap_file.get("num_slots", num_slots))
    if file_slots != num_slots:
        raise ValueError(f"Cap JSON num_slots={file_slots} differs from current Wave slots={num_slots}")
    file_timesteps = int(cap_file.get("num_train_timesteps", num_train_timesteps))
    if file_timesteps != num_train_timesteps:
        raise ValueError(
            f"Cap JSON num_train_timesteps={file_timesteps} differs from scheduler={num_train_timesteps}"
        )

    caps = cap_file.get("caps", cap_file)
    table = caps.get(region) if isinstance(caps, dict) else None
    if table is None:
        return None
    # Keep cap files on the public BrushNet slot schema (normally 28 rows).
    # Inactive topology rows are never used by the loss. To make future
    # Down+Mid cap builders robust, None/0/non-finite placeholders are accepted
    # only on inactive rows and replaced by 1.0 internally.
    if not isinstance(table, (list, tuple)) or len(table) != num_slots:
        raise ValueError(f"Cap table '{region}' must contain {num_slots} slot rows")
    active = set(range(num_slots) if active_slot_indices is None else map(int, active_slot_indices))
    cleaned = []
    for i, row in enumerate(table):
        if not isinstance(row, (list, tuple)) or len(row) != num_bins:
            raise ValueError(f"Cap table '{region}' row {i} must have {num_bins} bins")
        cleaned_row = []
        for value in row:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float('nan')
            if i in active:
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"Cap table '{region}' active slot {i} must contain finite positive values"
                    )
                cleaned_row.append(value)
            else:
                cleaned_row.append(value if math.isfinite(value) and value > 0 else 1.0)
        cleaned.append(cleaned_row)
    return torch.tensor(cleaned, dtype=torch.float32, device=device)


class WaveResidualEnvelopeLoss(nn.Module):
    """Global + known + hole one-sided per-slot residual envelopes.

    For region r and sample n / residual slot i:
        z = RMS_r(R_wave[n,i]) / C_r[i, bin(t_n)]
        phi = ReLU(z - beta_r)^2

    Region losses are weighted by per-slot s_i and normalized by the valid
    sample-slot weights. The total regularizer is the weighted sum of enabled
    region losses. C is fixed for the entire run: either a scalar override or a
    [num_slots, num_bins] table loaded once from JSON.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        num_train_timesteps: int,
        num_bins: int = 5,
        global_weight: float = 0.0,
        known_weight: float = 0.0,
        hole_weight: float = 0.0,
        global_beta: float = 0.95,
        known_beta: float = 0.90,
        hole_beta: float = 1.00,
        global_caps: Optional[torch.Tensor] = None,
        known_caps: Optional[torch.Tensor] = None,
        hole_caps: Optional[torch.Tensor] = None,
        global_slot_weights: Optional[torch.Tensor] = None,
        known_slot_weights: Optional[torch.Tensor] = None,
        hole_slot_weights: Optional[torch.Tensor] = None,
        active_slot_indices=None,
        min_region_fraction: float = 0.02,
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_slots < 1 or num_train_timesteps < 1 or num_bins < 1:
            raise ValueError("num_slots, num_train_timesteps and num_bins must be positive")
        if not 0 <= min_region_fraction < 1:
            raise ValueError("min_region_fraction must satisfy 0 <= value < 1")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.num_slots = int(num_slots)
        if active_slot_indices is None:
            active_slot_indices = list(range(self.num_slots))
        active_slot_indices = tuple(int(i) for i in active_slot_indices)
        if not active_slot_indices:
            raise ValueError('Residual envelope requires at least one active Wave slot')
        if len(set(active_slot_indices)) != len(active_slot_indices):
            raise ValueError('active_slot_indices must not contain duplicates')
        if min(active_slot_indices) < 0 or max(active_slot_indices) >= self.num_slots:
            raise ValueError('active_slot_indices contains an out-of-range residual slot')
        self.active_slot_indices = active_slot_indices
        self.active_slot_set = set(active_slot_indices)
        self.num_active_slots = len(active_slot_indices)
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_bins = int(num_bins)
        self.min_region_fraction = float(min_region_fraction)
        self.eps = float(eps)

        self.region_weights = {
            "global": float(global_weight),
            "known": float(known_weight),
            "hole": float(hole_weight),
        }
        self.betas = {
            "global": float(global_beta),
            "known": float(known_beta),
            "hole": float(hole_beta),
        }
        for region in REGIONS:
            if not torch.isfinite(torch.tensor(self.region_weights[region])) or self.region_weights[region] < 0:
                raise ValueError(f"{region} envelope weight must be finite and >= 0")
            if not torch.isfinite(torch.tensor(self.betas[region])) or not 0 <= self.betas[region] <= 1:
                raise ValueError(f"{region} beta must be finite and in [0,1]")

        cap_map = {"global": global_caps, "known": known_caps, "hole": hole_caps}
        slot_map = {
            "global": global_slot_weights,
            "known": known_slot_weights,
            "hole": hole_slot_weights,
        }
        for region in REGIONS:
            caps = cap_map[region]
            slots = slot_map[region]
            if self.region_weights[region] > 0 and caps is None:
                raise ValueError(
                    f"{region} residual-envelope loss is enabled but has no C values; "
                    f"set --wave_env_{region}_c or provide '{region}' in --wave_env_caps_json"
                )
            if caps is None:
                caps = torch.ones(self.num_slots, self.num_bins, dtype=torch.float32)
            if caps.shape != (self.num_slots, self.num_bins):
                raise ValueError(f"{region} caps have invalid shape {tuple(caps.shape)}")
            if slots is None:
                slots = torch.ones(self.num_slots, dtype=torch.float32, device=caps.device)
            if slots.shape != (self.num_slots,):
                raise ValueError(f"{region} slot weights have invalid shape {tuple(slots.shape)}")
            self.register_buffer(f"{region}_caps", caps.detach().float().clone(), persistent=True)
            self.register_buffer(f"{region}_slot_weights", slots.detach().float().clone(), persistent=True)

    @property
    def enabled(self) -> bool:
        return any(v > 0 for v in self.region_weights.values())

    def timestep_bins(self, timesteps: torch.Tensor) -> torch.Tensor:
        """High-noise timesteps map to bin 0; low-noise timesteps map to the last bin."""
        t = torch.as_tensor(timesteps).reshape(-1).long()
        t = t.clamp(0, self.num_train_timesteps - 1)
        reverse = (self.num_train_timesteps - 1) - t
        bins = torch.div(reverse * self.num_bins, self.num_train_timesteps, rounding_mode="floor")
        return bins.clamp(0, self.num_bins - 1)

    def _global_rms(self, residual: torch.Tensor):
        x = residual.float()
        rms = torch.sqrt(x.square().mean(dim=(1, 2, 3)) + self.eps)
        valid = torch.ones_like(rms, dtype=torch.bool)
        return rms, valid

    def _masked_rms(self, residual: torch.Tensor, region_mask: torch.Tensor):
        x = residual.float()
        m = F.interpolate(region_mask.float(), size=x.shape[-2:], mode="area").clamp_(0, 1)
        coverage = m.mean(dim=(1, 2, 3))
        valid = coverage >= self.min_region_fraction
        numerator = (x.square() * m).sum(dim=(1, 2, 3))
        denominator = x.shape[1] * m.sum(dim=(1, 2, 3))
        rms = torch.sqrt(numerator / denominator.clamp_min(self.eps) + self.eps)
        return rms, valid

    def _region_loss(self, region: str, residuals, hole: torch.Tensor, bins: torch.Tensor):
        first = residuals[self.active_slot_indices[0]]
        numerator = first.float().sum() * 0.0
        denominator = first.new_zeros((), dtype=torch.float32)
        valid_count = first.new_zeros((), dtype=torch.float32)
        active_count = first.new_zeros((), dtype=torch.float32)
        over_cap_count = first.new_zeros((), dtype=torch.float32)
        z_sum = first.new_zeros((), dtype=torch.float32)
        rms_sum = first.new_zeros((), dtype=torch.float32)
        z_max = first.new_zeros((), dtype=torch.float32)
        rms_max = first.new_zeros((), dtype=torch.float32)

        caps = getattr(self, f"{region}_caps")
        slot_weights = getattr(self, f"{region}_slot_weights")
        beta = self.betas[region]
        region_mask = None
        if region == "known":
            region_mask = 1.0 - hole.float()
        elif region == "hole":
            region_mask = hole.float()

        for i in self.active_slot_indices:
            residual = residuals[i]
            if residual.shape[0] != bins.shape[0]:
                raise ValueError(
                    f"Residual batch {residual.shape[0]} differs from timestep batch {bins.shape[0]} at slot {i}"
                )
            if region == "global":
                rms, valid = self._global_rms(residual)
            else:
                rms, valid = self._masked_rms(residual, region_mask)

            cap = caps[i].index_select(0, bins.to(caps.device)).to(device=rms.device, dtype=rms.dtype)
            z = rms / cap.clamp_min(self.eps)
            penalty = F.relu(z - beta).square()
            valid_f = valid.to(dtype=penalty.dtype)
            slot_w = slot_weights[i].to(device=penalty.device, dtype=penalty.dtype)

            numerator = numerator + slot_w * (penalty * valid_f).sum()
            denominator = denominator + slot_w * valid_f.sum()
            valid_count = valid_count + valid_f.sum()
            active_count = active_count + ((z > beta) & valid).float().sum()
            over_cap_count = over_cap_count + ((z > 1.0) & valid).float().sum()
            z_sum = z_sum + (z * valid_f).sum()
            rms_sum = rms_sum + (rms * valid_f).sum()
            if bool(valid.any()):
                z_max = torch.maximum(z_max, z.masked_fill(~valid, 0).max())
                rms_max = torch.maximum(rms_max, rms.masked_fill(~valid, 0).max())

        loss = numerator / denominator.clamp_min(self.eps)
        count = valid_count.clamp_min(1.0)
        stats = {
            "loss": loss,
            "valid_fraction": valid_count / float(bins.shape[0] * self.num_active_slots),
            "active_fraction": active_count / count,
            "over_cap_fraction": over_cap_count / count,
            "z_mean": z_sum / count,
            "z_max": z_max,
            "rms_mean": rms_sum / count,
            "rms_max": rms_max,
        }
        return loss, stats

    def forward(self, residuals, hole: torch.Tensor, timesteps: torch.Tensor) -> Dict[str, torch.Tensor]:
        if len(residuals) != self.num_slots:
            raise ValueError(f"Expected {self.num_slots} Wave residuals, got {len(residuals)}")
        batch = residuals[0].shape[0]
        if hole.ndim != 4 or hole.shape[0] != batch or hole.shape[1] != 1:
            raise ValueError(f"Expected hole mask [B,1,H,W] with B={batch}, got {tuple(hole.shape)}")
        t = torch.as_tensor(timesteps, device=residuals[0].device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if t.numel() != batch:
            raise ValueError(f"Timestep batch {t.numel()} differs from residual batch {batch}")
        bins = self.timestep_bins(t).to(residuals[0].device)

        total = residuals[self.active_slot_indices[0]].float().sum() * 0.0
        output: Dict[str, torch.Tensor] = {"total": total}
        for region in REGIONS:
            weight = self.region_weights[region]
            if weight <= 0:
                continue
            region_loss, stats = self._region_loss(region, residuals, hole, bins)
            total = total + weight * region_loss
            output[f"{region}_weighted"] = weight * region_loss
            for key, value in stats.items():
                output[f"{region}_{key}"] = value
        output["total"] = total
        return output

    def export_config(self) -> dict:
        out = {
            "num_slots": self.num_slots,
            "num_active_slots": self.num_active_slots,
            "active_slot_indices": list(self.active_slot_indices),
            "num_bins": self.num_bins,
            "num_train_timesteps": self.num_train_timesteps,
            "min_region_fraction": self.min_region_fraction,
            "region_weights": self.region_weights,
            "betas": self.betas,
            "caps": {},
            "slot_weights": {},
        }
        for region in REGIONS:
            out["caps"][region] = getattr(self, f"{region}_caps").detach().cpu().tolist()
            out["slot_weights"][region] = getattr(self, f"{region}_slot_weights").detach().cpu().tolist()
        return out


def build_wave_residual_envelope(args, wave, num_train_timesteps: int, device):
    """Build the fixed envelope regularizer from CLI args.

    Region-specific fixed C overrides the corresponding JSON table. A disabled
    region (weight=0) does not require a cap.
    """
    weights = {
        "global": float(args.wave_env_global_weight),
        "known": float(args.wave_env_known_weight),
        "hole": float(args.wave_env_hole_weight),
    }
    if not any(v > 0 for v in weights.values()):
        return None
    if wave is None:
        raise ValueError("Residual-envelope regularization requires WaveConditioner")

    num_slots = len(wave.spec["slots"])
    active_slot_indices = list(getattr(wave, 'active_slot_indices', range(num_slots)))
    num_bins = int(args.wave_env_bins)
    cap_file = _load_cap_file(args.wave_env_caps_json)

    caps = {}
    slot_weights = {}
    for region in REGIONS:
        caps[region] = _cap_table(
            region,
            getattr(args, f"wave_env_{region}_c"),
            cap_file,
            num_slots,
            num_bins,
            num_train_timesteps,
            device,
            active_slot_indices=active_slot_indices,
        )
        slot_weights[region] = _as_slot_weights(
            getattr(args, f"wave_env_{region}_slot_weights"), num_slots, device
        )

    return WaveResidualEnvelopeLoss(
        num_slots=num_slots,
        num_train_timesteps=num_train_timesteps,
        num_bins=num_bins,
        global_weight=weights["global"],
        known_weight=weights["known"],
        hole_weight=weights["hole"],
        global_beta=float(args.wave_env_global_beta),
        known_beta=float(args.wave_env_known_beta),
        hole_beta=float(args.wave_env_hole_beta),
        global_caps=caps["global"],
        known_caps=caps["known"],
        hole_caps=caps["hole"],
        global_slot_weights=slot_weights["global"],
        known_slot_weights=slot_weights["known"],
        hole_slot_weights=slot_weights["hole"],
        active_slot_indices=active_slot_indices,
        min_region_fraction=float(args.wave_env_min_region_fraction),
    ).to(device)
