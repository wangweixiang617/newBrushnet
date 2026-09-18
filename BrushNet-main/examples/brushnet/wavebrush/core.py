"""MVWT + four adapters + time gates; all public masks use 1=hole."""
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

BANDS = ('H1', 'H2', 'H3', 'L3')
PRESETS = {
    'B1_rgb': dict(transform='rgb', reliability='none', gate='fixed'),
    'B2_dwt': dict(transform='dwt', reliability='none', gate='fixed'),
    'B3_mvwt': dict(transform='mvwt', reliability='none', gate='fixed'),
    'B4_concat': dict(transform='mvwt', reliability='concat', gate='fixed'),
    'B5_premul': dict(transform='mvwt', reliability='premul', gate='fixed'),
    'B6_constant': dict(transform='mvwt', reliability='premul', gate='constant'),
    'B7_time': dict(transform='mvwt', reliability='premul', gate='time'),
    'recommended': dict(transform='mvwt', reliability='concat', gate='time'),
    'soft_time': dict(transform='mvwt', reliability='soft', gate='time'),
}


def haar_step(x):
    a, b, c, d = x[..., ::2, ::2], x[..., ::2, 1::2], x[..., 1::2, ::2], x[..., 1::2, 1::2]
    return (a+b+c+d)/2, torch.cat(((-a-b+c+d)/2, (-a+b-c+d)/2, (a-b-c+d)/2), 1)


def support_step(c):
    a, b, d, e = c[..., ::2, ::2], c[..., ::2, 1::2], c[..., 1::2, ::2], c[..., 1::2, 1::2]
    harmonic = lambda x, y: 2*x*y/(x+y).clamp_min(1e-12)
    low = (a+b+d+e)/4
    high = torch.cat((harmonic((a+b)/2, (d+e)/2), harmonic((a+d)/2, (b+e)/2),
                      harmonic((a+e)/2, (b+d)/2)), 1)
    return low, high


def haar_pyramid(image, hole, transform='mvwt', support='directional'):
    """image: RGB [-1,1], hole: B1HW in [0,1]; unobserved pixels removed internally.
    High-band channel order: LH(RGB), HL(RGB), HH(RGB). Exactly three levels.
    """
    if image.ndim != 4 or image.shape[1] != 3 or hole.shape != image[:, :1].shape:
        raise ValueError('Expected image B3HW and hole B1HW')
    if image.shape[-1] % 8 or image.shape[-2] % 8:
        raise ValueError('H and W must be divisible by 8')
    if transform not in ('mvwt', 'dwt') or support not in ('directional', 'average', 'hard'):
        raise ValueError('Unknown transform/support')
    # Arithmetic stays float32, including under autocast and half inference.
    x, c = image.float() * (1-hole.float()), 1-hole.float()
    bands, qs = [], []
    for _ in range(3):
        if transform == 'mvwt':
            total = F.avg_pool2d(c, 2)*4
            weighted = F.avg_pool2d(x*c, 2)*4
            mu = weighted/total.clamp_min(1e-12)
            mu = mu.repeat_interleave(2, -2).repeat_interleave(2, -1)
            x = c*x + (1-c)*mu
        low_q, high_q = support_step(c)
        x, high = haar_step(x)
        if support == 'average':
            high_q = low_q.repeat(1, 3, 1, 1)
        elif support == 'hard':
            high_q = (high_q >= 1-1e-6).float()
        bands.append(high)
        qs.append(high_q)
        c = low_q
    bands.append(x)
    qs.append((c >= 1-1e-6).float() if support == 'hard' else c)
    return bands, qs


def flatten_residuals(result):
    if isinstance(result, (tuple, list)):
        down, mid, up = result
    else:
        down, mid, up = result.down_block_res_samples, result.mid_block_res_sample, result.up_block_res_samples
    return list(down) + [mid] + list(up), len(down), len(up)


def slot_spec(result, latent_hw):
    samples, nd, nu = flatten_residuals(result)
    slots = []
    for x in samples:
        ratio = latent_hw[0] // x.shape[-2]
        scale = int(round(math.log2(ratio)))
        if scale not in range(4) or tuple(x.shape[-2:]) != tuple(v//(2**scale) for v in latent_hw):
            raise ValueError(f'Unsupported residual shape {tuple(x.shape)}, latent={latent_hw}')
        slots.append(dict(channels=x.shape[1], scale=scale))
    return dict(slots=slots, n_down=nd, n_up=nu)


def merge_residuals(base, additions, strength=1.):
    samples, nd, nu = flatten_residuals(base)
    if len(samples) != len(additions):
        raise ValueError('Residual slot count differs')
    out = []
    for x, y in zip(samples, additions):
        if x.shape != y.shape:
            raise ValueError(f'Residual shape mismatch: {x.shape} vs {y.shape}')
        out.append(x + strength*y.to(x.dtype))
    return tuple(out[:nd]), out[nd], tuple(out[nd+1:])


class Adapter(nn.Module):
    def __init__(self, channels, widths):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(channels, widths[0], 1), nn.SiLU(), nn.Conv2d(widths[0], widths[0], 3, padding=1))
        self.down = nn.ModuleList(nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4*a, b, 1), nn.SiLU(),
                                                nn.Conv2d(b, b, 3, padding=1)) for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, x):
        x = self.stem(x)
        out = [x]
        for block in self.down:
            x = block(x)
            out.append(x)
        return out


class Gates(nn.Module):
    def __init__(self, mode='time', maximum=1., initial=.5, timesteps=1000):
        super().__init__()
        if mode not in ('fixed', 'constant', 'time') or not 0 < initial < maximum:
            raise ValueError('Invalid gate configuration')
        self.mode, self.maximum, self.initial, self.timesteps = mode, maximum, initial, timesteps
        self.register_buffer('frequencies', torch.exp(-math.log(10000)*torch.arange(32)/32))
        logit = math.log(initial/(maximum-initial))
        if mode == 'constant':
            self.logits = nn.Parameter(torch.full((4, 4), logit))
        elif mode == 'time':
            ## 扩充了网络结构，参考unet的 timeStepEmbedding *4
            self.net = nn.Sequential(nn.Linear(64, 64 * 4), nn.SiLU(), nn.Linear(64 * 4, 16))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, logit)

    def forward(self, t):
        if self.mode == 'fixed':
            return torch.full((len(t), 4, 4), self.initial, device=t.device)
        if self.mode == 'constant':
            return self.maximum*self.logits.sigmoid()[None].expand(len(t), -1, -1)
        freq = torch.exp(-math.log(10000)*torch.arange(32,device=t.device,dtype=torch.float32)/32)
        phase = t.float()[:, None] * (1000./self.timesteps) * freq[None]
        emb = torch.cat((phase.cos(), phase.sin()), 1).to(self.net[0].weight.dtype)
        return self.maximum*self.net(emb).reshape(-1, 4, 4).sigmoid()


class WaveConditioner(nn.Module):
    def __init__(self, spec, transform='mvwt', reliability='concat', gate='time', support='directional',
                 widths=(32,64,96,128), gate_max=1., gate_init=.5, timesteps=1000, soft_lambda=.5,
                 shared=False, coarse_only=False):
        super().__init__()
        if reliability not in ('none', 'concat', 'premul', 'soft'):
            raise ValueError('Unknown reliability mode')
        if transform not in ('mvwt', 'dwt', 'rgb'):
            raise ValueError('Unknown transform')
        if not 0 <= soft_lambda <= 1 or len(widths) != 4 or min(widths) < 1:
            raise ValueError('Invalid soft_lambda/adapter widths')
        self.config = dict(spec=spec, transform=transform, reliability=reliability, gate=gate, support=support,
                           widths=list(widths), gate_max=gate_max, gate_init=gate_init, timesteps=timesteps,
                           soft_lambda=soft_lambda, shared=shared, coarse_only=coarse_only)
        self.spec = spec
        # Non-gradient RMS buffers; EMA updates occur only after successful optimizer steps.
        self.register_buffer('band_rms', torch.ones(10))
        self.register_buffer('rms_fitted', torch.tensor(False))
        self.register_buffer('rms_second_moment', torch.ones(10))
        self.register_buffer('rms_updates', torch.tensor(0, dtype=torch.long))
        # Keep q channels even in no-q controls, set them to zero. Parameter count stays identical B2-B7.
        in_channels = [192,48,12,4] if transform != 'rgb' else [192,192,192,192]
        if shared:
            self.projections = nn.ModuleList(nn.Conv2d(c, widths[0], 1) for c in in_channels)
            self.adapters = nn.ModuleList([Adapter(widths[0], widths)])
        else:
            self.projections = nn.ModuleList()
            self.adapters = nn.ModuleList(Adapter(c, widths) for c in in_channels)
        self.gates = Gates(gate, gate_max, gate_init, timesteps)
        self.zero = nn.ModuleList(nn.Conv2d(4*widths[s['scale']], s['channels'], 1, bias=False) for s in spec['slots'])
        for layer in self.zero:
            nn.init.zeros_(layer.weight)
        # Evaluation-only interventions; persisted configurations remain unchanged.
        self.drop_bands = set()
        self.drop_scales = set()
        self.drop_interval = None  # normalized forward t in [0,1]; large t is early denoising
        self.gate_override = None
        self.strength = 1.

    def _apply(self, fn):
        super()._apply(fn)
        self.band_rms = self.band_rms.float()
        self.rms_second_moment = self.rms_second_moment.float()
        return self

    def set_rms(self, path):
        data = json.loads(Path(path).read_text())
        if data.get('image_range') != '[-1,1]' or data.get('order') != ['H1_LH','H1_HL','H1_HH','H2_LH','H2_HL','H2_HH','H3_LH','H3_HL','H3_HH','L3']:
            raise ValueError('RMS file preprocessing/order differs')
        values = torch.tensor(data['rms'], device=self.band_rms.device)
        if values.shape != (10,) or not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError('RMS values must be 10 finite positive numbers')
        self.band_rms.copy_(values)
        self.rms_second_moment.copy_(values.square())
        self.rms_fitted.fill_(True)

    def encode(self, image, hole):
        if image.shape[-1] % 64 or image.shape[-2] % 64:
            raise ValueError('Image H/W must be divisible by 64 for four UNet scales')
        cfg = self.config
        dtype = next(self.adapters.parameters()).dtype
        if cfg['transform'] == 'rgb':
            rgb = F.pixel_unshuffle(image.float()*(1-hole.float()), 8).to(dtype)
            inputs = [rgb]*4
        else:
            bands, qs = haar_pyramid(image, hole, cfg['transform'], cfg['support'])
            inputs = []
            for i, (w, q, r) in enumerate(zip(bands, qs, (4,2,1,1))):
                rms = self.band_rms[i*3:i*3+3] if i < 3 else self.band_rms[9:10]
                w = w/rms.repeat_interleave(3)[None,:,None,None].clamp_min(1e-6)
                if cfg['reliability'] == 'premul':
                    w = w*q.repeat_interleave(3, 1)
                elif cfg['reliability'] == 'soft':
                    w = w*(1-cfg['soft_lambda']*(1-q.repeat_interleave(3, 1)))
                q_input = torch.zeros_like(q) if cfg['reliability'] == 'none' else q
                x = torch.cat((w,q_input), 1)
                if cfg['coarse_only'] and i < 2:
                    x = torch.zeros_like(x)
                inputs.append(F.pixel_unshuffle(x,r).to(dtype))
        if cfg['shared']:
            features = [self.adapters[0](proj(x)) for proj,x in zip(self.projections,inputs)]
        else:
            features = [adapter(x) for adapter,x in zip(self.adapters,inputs)]
        if cfg['coarse_only']:
            features[:2] = [[f*0 for f in branch] for branch in features[:2]]
        return features

    def effective_gates(self, t):
        batch = len(t)
        g = self.gates(t)
        if self.gate_override is not None:
            g = self.gate_override.to(g).reshape(1,4,4).expand(batch,-1,-1)
        # If both a band and time window are selected, drop only that band IN that window.
        selected = torch.ones_like(g, dtype=torch.bool)
        if self.drop_bands:
            selected.zero_()
            for b in self.drop_bands:
                selected[:,b,:] = True
        if self.drop_scales:
            scales = torch.zeros_like(selected)
            for scale in self.drop_scales:
                scales[:,:,scale] = True
            selected &= scales
        if self.drop_interval is not None:
            lo, hi = self.drop_interval
            active = (t/(self.config['timesteps']-1) >= lo) & (t/(self.config['timesteps']-1) <= hi)
            selected &= active[:,None,None]
        elif not self.drop_bands and not self.drop_scales:
            selected.zero_()
        return g * (~selected).to(g.dtype)

    def project(self, features, timesteps):
        batch = features[0][0].shape[0]
        t = torch.as_tensor(timesteps, device=features[0][0].device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if len(t) != batch:
            raise ValueError('Timestep batch mismatch')
        g = self.effective_gates(t)
        fused = [torch.cat([features[b][s]*g[:,b,s,None,None,None].to(features[b][s].dtype) for b in range(4)],1) for s in range(4)]
        return [self.strength*layer(fused[slot['scale']]) for layer,slot in zip(self.zero,self.spec['slots'])]

    @torch.no_grad()
    def branch_rms(self, features, timesteps):
        """Actual projected per-band contribution; sums reconstruct the full wave residual."""
        batch = features[0][0].shape[0]
        t = torch.as_tensor(timesteps, device=features[0][0].device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        g = self.effective_gates(t)
        rows = []
        for layer, slot in zip(self.zero, self.spec['slots']):
            scale = slot['scale']
            width = features[0][scale].shape[1]
            values = []
            for b in range(4):
                x = features[b][scale]*g[:,b,scale,None,None,None].to(features[b][scale].dtype)
                weight = layer.weight[:,b*width:(b+1)*width]
                z = self.strength*F.conv2d(x,weight)
                values.append(float(z.float().square().mean().sqrt()))
            rows.append(values)
        return rows

    def forward(self, image, hole, timesteps):
        return self.project(self.encode(image,hole),timesteps)

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True,exist_ok=True)
        (path/'config.json').write_text(json.dumps(self.config,indent=2))
        torch.save(self.state_dict(),path/'weights.pt')

    @classmethod
    def from_pretrained(cls,path,device='cpu'):
        path = Path(path)
        obj = cls(**json.loads((path/'config.json').read_text()))
        state = torch.load(path/'weights.pt',map_location='cpu',weights_only=True)
        state.setdefault('rms_second_moment', state['band_rms'].float().square())
        state.setdefault('rms_updates', torch.tensor(0, dtype=torch.long))
        obj.load_state_dict(state)
        return obj.to(device)
