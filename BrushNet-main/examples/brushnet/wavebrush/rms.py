"""RMS statistics: ordinary DWT of full TRAIN images, never validation images."""
import torch
import torch.distributed as dist
from .core import haar_pyramid

ORDER = ['H1_LH','H1_HL','H1_HH','H2_LH','H2_HL','H2_HH','H3_LH','H3_HL','H3_HH','L3']


@torch.no_grad()
def sufficient_statistics(images):
    bands, _ = haar_pyramid(images.float(), torch.zeros_like(images[:, :1]), 'dwt')
    groups = [part for band in bands[:3] for part in band.split(3, 1)] + [bands[3]]
    sums = torch.stack([w.double().square().sum() for w in groups])
    counts = torch.tensor([w.numel() for w in groups], device=images.device, dtype=torch.float64)
    return torch.stack((sums, counts))


class RMSAccumulator:
    def __init__(self, device, mode='ema', decay=.99):
        if mode not in ('ema', 'fixed') or not 0 <= decay < 1:
            raise ValueError('Invalid RMS mode/decay')
        self.mode, self.decay = mode, decay
        self.pending = torch.zeros(2, 10, device=device, dtype=torch.float64)

    @torch.no_grad()
    def add(self, images):
        if self.mode == 'ema':
            self.pending.add_(sufficient_statistics(images))

    @torch.no_grad()
    def finish(self, wave, successful=True):
        if self.mode == 'ema' and successful:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(self.pending, op=dist.ReduceOp.SUM)
            sums, counts = self.pending
            if (counts <= 0).any() or not torch.isfinite(sums).all():
                raise ValueError('Invalid distributed RMS statistics')
            moment = (sums / counts).clamp_min(1e-8).float()
            wave.rms_second_moment.mul_(self.decay).add_(moment, alpha=1-self.decay)
            wave.band_rms.copy_(wave.rms_second_moment.sqrt().clamp_min(1e-4))
            wave.rms_updates.add_(1)
            wave.rms_fitted.fill_(True)
        self.pending.zero_()


@torch.no_grad()
def initialize_rms_from_loader(wave, dataloader, accelerator, num_batches=4):
    """Warm up from the ORIGINAL prepared streaming loader; all ranks participate.
    This is an extra short pass, not an optimizer step. Training starts a fresh iterator.
    No len(), indexing, SQLite, or replacement Dataset is required.
    """
    import random
    import numpy as np
    if wave is None:
        return
    raw = accelerator.unwrap_model(wave)
    if raw.config['transform'] == 'rgb':
        return
    fitted = torch.tensor([int(raw.rms_fitted.item())], device=accelerator.device)
    count = int(accelerator.reduce(fitted, reduction='sum').item())
    if count == accelerator.num_processes:
        accelerator.print('[RMS] Loaded saved RMS; initialization skipped.')
        return
    if count != 0:
        raise RuntimeError('RMS initialized state differs across ranks')
    if num_batches < 1:
        raise ValueError('rms_init_batches must be positive')
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(accelerator.device) if accelerator.device.type == 'cuda' else None
    generators = {}
    for name in ('generator', 'synchronized_generator'):
        g = getattr(dataloader, name, None)
        if isinstance(g, torch.Generator):
            generators[id(g)] = (g, g.get_state())
    iteration = getattr(dataloader, 'iteration', None)
    total = torch.zeros(2, 10, device=accelerator.device, dtype=torch.float64)
    images_seen = torch.zeros(1, device=accelerator.device, dtype=torch.long)
    iterator = None
    try:
        iterator = iter(dataloader)
        for i in range(num_batches):
            error = None
            try:
                batch = next(iterator)
                images = batch['pixel_values'].to(accelerator.device, dtype=torch.float32)
                statistics = sufficient_statistics(images)
                if not torch.isfinite(statistics).all():
                    raise ValueError('nonfinite RMS input statistics')
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            failed = torch.tensor([int(error is not None)], device=accelerator.device)
            if accelerator.reduce(failed, reduction='sum').item():
                raise RuntimeError(f'RMS warmup failed on one or more ranks; local error: {error}')
            total.add_(statistics)
            images_seen.add_(images.shape[0])
            accelerator.print(f'[RMS] warmup local batch {i+1}/{num_batches}')
        total = accelerator.reduce(total, reduction='sum')
        images_seen = accelerator.reduce(images_seen, reduction='sum')
        moment = (total[0] / total[1]).clamp_min(1e-8).float()
        raw.rms_second_moment.copy_(moment)
        raw.band_rms.copy_(moment.sqrt().clamp_min(1e-4))
        raw.rms_fitted.fill_(True)
        raw.rms_updates.zero_()
        accelerator.print(f'[RMS] initialized from {images_seen.item()} images: {raw.band_rms.cpu().tolist()}')
    finally:
        if iterator is not None:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
            del iterator
        # An early break in an Accelerate iterator may not execute its normal end().
        state = getattr(dataloader, 'gradient_state', None)
        if state is not None and state.active_dataloader is dataloader:
            dataloader.end()
        if iteration is not None:
            dataloader.set_epoch(iteration)
        for g, state in generators.values():
            g.set_state(state)
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, accelerator.device)
