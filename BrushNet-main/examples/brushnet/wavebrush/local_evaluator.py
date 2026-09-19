"""Rank-0 validation adapter; original validation_evaluator.py is unchanged."""
from ..validation_evaluator import BrushNetValidationEvaluator as OriginalEvaluator


def _no_distributed():
    return False


def local_only(metric):
    # TorchMetrics must not enter collectives while the other ranks await validation.
    metric.sync_on_compute = False
    metric.dist_sync_on_step = False
    # Some versions cache sync_on_compute internally; disable distributed availability too.
    metric.distributed_available_fn = _no_distributed


class BrushNetValidationEvaluator(OriginalEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        local_only(self.lpips_metric)

    def _load_clip_metric(self):
        super()._load_clip_metric()
        local_only(self.clip_metric)
