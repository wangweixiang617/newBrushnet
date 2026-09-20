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
    'B2_dwt': dict(transform='dwt', reliability='concat', gate='fixed'),
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
    return list(out[:nd]), out[nd], list(out[nd+1:])


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


class UNetAdapter(nn.Module):
    """UNet-style adapter: external PixelUnshuffle -> Conv -> 4 ResNet stages + 3 Downsample2D.

    Input/output contract is identical to Adapter: input is already 64x64 and forward returns
    four feature maps at 64/32/16/8 with channels given by widths.
    """
    def __init__(self, channels, widths):
        super().__init__()
        try:
            from diffusers.models.unets.unet_2d_blocks import DownBlock2D
            try:
                from diffusers.models.downsampling import Downsample2D
            except ImportError:
                from diffusers.models.resnet import Downsample2D
        except ImportError as exc:
            raise ImportError('UNetAdapter requires the same diffusers package used by BrushNet') from exc

        def groups(a, b):
            # GroupNorm group count must divide both input/output channels of the first ResNet.
            g = min(32, math.gcd(a, b))
            while g > 1 and (a % g or b % g):
                g -= 1
            return g

        # Band-specific stem. When target=64, H1 is exactly 192->128->64,
        # while H2/H3/L3 use a direct projection to 64.
        target = widths[0]
        if channels > 2 * target:
            hidden = 2 * target
            self.stem = nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, target, 3, padding=1),
            )
        else:
            self.stem = nn.Conv2d(channels, target, 3, padding=1)

        self.blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = widths[0]
        for i, out_ch in enumerate(widths):
            self.blocks.append(DownBlock2D(in_channels=in_ch, out_channels=out_ch, temb_channels=None, num_layers=2,
                                           add_downsample=False, resnet_eps=1e-5, resnet_act_fn='silu',
                                           resnet_groups=groups(in_ch, out_ch)))
            if i < len(widths)-1:
                self.downsamplers.append(Downsample2D(out_ch, use_conv=True, out_channels=out_ch, padding=1, name='op'))
            in_ch = out_ch

    def forward(self, x):
        x = self.stem(x)
        out = []
        for i, block in enumerate(self.blocks):
            x, _ = block(x, temb=None)
            out.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
        return out


class SelfAttnUNetAdapter(nn.Module):
    """UNet-style adapter with direct spatial self-attention and a band-specific stem.

    Input is already aligned to 64x64 by WaveConditioner PixelUnshuffle.
    For widths[0]=64 and the standard MVWT inputs, stems are:
      H1: 192 -> 128 -> 64
      H2:  48 -> 64
      H3:  12 -> 64
      L3:   4 -> 64
    The adapter returns four feature maps at 64/32/16/8 with channels=widths.
    No encoder_hidden_states and no extra timestep embedding are required.
    """
    def __init__(self, channels, widths, attention_head_dim=8):
        super().__init__()
        try:
            from diffusers.models.unets.unet_2d_blocks import AttnDownBlock2D
            try:
                from diffusers.models.downsampling import Downsample2D
            except ImportError:
                from diffusers.models.resnet import Downsample2D
        except ImportError as exc:
            raise ImportError('SelfAttnUNetAdapter requires the same diffusers package used by the UNet') from exc

        if attention_head_dim < 1:
            raise ValueError('attention_head_dim must be positive')
        if any(w % attention_head_dim for w in widths):
            raise ValueError('Every adapter width must be divisible by attention_head_dim')

        def groups(a, b):
            g = min(32, math.gcd(a, b))
            while g > 1 and (a % g or b % g):
                g -= 1
            return g

        # Band-specific stem. When target=64, H1 is exactly 192->128->64,
        # while H2/H3/L3 use a direct projection to 64.
        target = widths[0]
        if channels > 2 * target:
            hidden = 2 * target
            self.stem = nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, target, 3, padding=1),
            )
        else:
            self.stem = nn.Conv2d(channels, target, 3, padding=1)

        self.blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = target
        for i, out_ch in enumerate(widths):
            self.blocks.append(
                AttnDownBlock2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=None,
                    num_layers=2,
                    resnet_eps=1e-5,
                    resnet_act_fn='silu',
                    resnet_groups=groups(in_ch, out_ch),
                    attention_head_dim=attention_head_dim,
                    downsample_type=None,
                )
            )
            if i < len(widths)-1:
                self.downsamplers.append(
                    Downsample2D(out_ch, use_conv=True, out_channels=out_ch, padding=1, name='op')
                )
            in_ch = out_ch

    def forward(self, x):
        x = self.stem(x)
        out = []
        for i, block in enumerate(self.blocks):
            x, _ = block(hidden_states=x, temb=None)
            out.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
        return out


class HybridSelfAttnUNetAdapter(nn.Module):
    """Hybrid UNet adapter.

    Keeps the existing band-specific stem, then uses:
      64x64: DownBlock2D(num_layers=2)
      32x32: DownBlock2D(num_layers=2)
      16x16: AttnDownBlock2D(num_layers=2)
       8x8 : AttnDownBlock2D(num_layers=2)

    Existing SelfAttnUNetAdapter and CrossAttnUNetAdapter are left unchanged.
    """
    def __init__(self, channels, widths, attention_head_dim=8):
        super().__init__()
        try:
            from diffusers.models.unets.unet_2d_blocks import DownBlock2D, AttnDownBlock2D
            try:
                from diffusers.models.downsampling import Downsample2D
            except ImportError:
                from diffusers.models.resnet import Downsample2D
        except ImportError as exc:
            raise ImportError(
                'HybridSelfAttnUNetAdapter requires the same diffusers package used by the UNet'
            ) from exc

        if attention_head_dim < 1:
            raise ValueError('attention_head_dim must be positive')
        if any(w % attention_head_dim for w in widths[2:]):
            raise ValueError(
                'Hybrid attention-stage widths (16x16/8x8) must be divisible by attention_head_dim'
            )

        def groups(a, b):
            g = min(32, math.gcd(a, b))
            while g > 1 and (a % g or b % g):
                g -= 1
            return g

        # Same band-specific stem rule as the uploaded UNetAdapter,
        # SelfAttnUNetAdapter and CrossAttnUNetAdapter.
        # With widths[0]=64:
        #   H1: 192 -> 128 -> 64
        #   H2:  48 -> 64
        #   H3:  12 -> 64
        #   L3:   4 -> 64
        target = widths[0]
        if channels > 2 * target:
            hidden = 2 * target
            self.stem = nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, target, 3, padding=1),
            )
        else:
            self.stem = nn.Conv2d(channels, target, 3, padding=1)

        self.blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        in_ch = target
        for i, out_ch in enumerate(widths):
            if i < 2:
                block = DownBlock2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=None,
                    num_layers=2,
                    add_downsample=False,
                    resnet_eps=1e-5,
                    resnet_act_fn='silu',
                    resnet_groups=groups(in_ch, out_ch),
                )
            else:
                block = AttnDownBlock2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=None,
                    num_layers=2,
                    resnet_eps=1e-5,
                    resnet_act_fn='silu',
                    resnet_groups=groups(in_ch, out_ch),
                    attention_head_dim=attention_head_dim,
                    downsample_type=None,
                )

            self.blocks.append(block)

            if i < len(widths) - 1:
                self.downsamplers.append(
                    Downsample2D(
                        out_ch,
                        use_conv=True,
                        out_channels=out_ch,
                        padding=1,
                        name='op',
                    )
                )
            in_ch = out_ch

    def forward(self, x):
        x = self.stem(x)
        out = []

        for i, block in enumerate(self.blocks):
            x, _ = block(hidden_states=x, temb=None)
            out.append(x)

            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)

        return out


class CrossAttnUNetAdapter(nn.Module):
    """UNet-style adapter using CrossAttnDownBlock2D at each scale.

    The external WaveConditioner still aligns all wave bands to 64x64 before this module.
    This adapter keeps the same output contract as Adapter/UNetAdapter: four feature maps
    at 64/32/16/8. It reuses the already-computed text encoder hidden states and does not
    create a separate timestep embedding (temb_channels=None).
    """
    def __init__(self, channels, widths, cross_attention_dim=768, num_attention_heads=8):
        super().__init__()
        try:
            from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D
            try:
                from diffusers.models.downsampling import Downsample2D
            except ImportError:
                from diffusers.models.resnet import Downsample2D
        except ImportError as exc:
            raise ImportError('CrossAttnUNetAdapter requires the same diffusers package used by the UNet') from exc

        if cross_attention_dim < 1 or num_attention_heads < 1:
            raise ValueError('Invalid cross-attention configuration')
        if any(w % num_attention_heads for w in widths):
            raise ValueError('Every adapter width must be divisible by num_attention_heads')

        def groups(a, b):
            g = min(32, math.gcd(a, b))
            while g > 1 and (a % g or b % g):
                g -= 1
            return g

        self.cross_attention_dim = cross_attention_dim

        # Band-specific stem. When target=64, H1 is exactly 192->128->64,
        # while H2/H3/L3 use a direct projection to 64.
        target = widths[0]
        if channels > 2 * target:
            hidden = 2 * target
            self.stem = nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, target, 3, padding=1),
            )
        else:
            self.stem = nn.Conv2d(channels, target, 3, padding=1)

        self.blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = widths[0]
        for i, out_ch in enumerate(widths):
            self.blocks.append(
                CrossAttnDownBlock2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=None,
                    num_layers=2,
                    transformer_layers_per_block=1,
                    add_downsample=False,
                    resnet_eps=1e-5,
                    resnet_act_fn='silu',
                    resnet_groups=groups(in_ch, out_ch),
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=num_attention_heads,
                    dual_cross_attention=False,
                    use_linear_projection=False,
                    only_cross_attention=False,
                    upcast_attention=False,
                    attention_type='default',
                )
            )
            if i < len(widths)-1:
                self.downsamplers.append(
                    Downsample2D(out_ch, use_conv=True, out_channels=out_ch, padding=1, name='op')
                )
            in_ch = out_ch

    def forward(self, x, encoder_hidden_states):
        if encoder_hidden_states is None:
            raise ValueError('CrossAttnUNetAdapter requires encoder_hidden_states')
        if encoder_hidden_states.shape[-1] != self.cross_attention_dim:
            raise ValueError(
                f'encoder_hidden_states dim {encoder_hidden_states.shape[-1]} != cross_attention_dim {self.cross_attention_dim}'
            )
        if encoder_hidden_states.shape[0] != x.shape[0]:
            raise ValueError(
                f'encoder_hidden_states batch {encoder_hidden_states.shape[0]} != feature batch {x.shape[0]}'
            )

        x = self.stem(x)
        encoder_hidden_states = encoder_hidden_states.to(device=x.device, dtype=x.dtype)
        out = []
        for i, block in enumerate(self.blocks):
            x, _ = block(
                hidden_states=x,
                temb=None,
                encoder_hidden_states=encoder_hidden_states,
            )
            out.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
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
                 shared=False, coarse_only=False, adapter_type='legacy', cross_attention_dim=768,
                 cross_attention_heads=8, drop_bands=None, drop_scales=None, drop_interval=None,
                 gate_override=None, wave_strength=1.):
        super().__init__()
        if reliability not in ('none', 'concat', 'premul', 'soft'):
            raise ValueError('Unknown reliability mode')
        if transform not in ('mvwt', 'dwt', 'rgb'):
            raise ValueError('Unknown transform')
        if adapter_type not in ('legacy', 'unet', 'selfattn', 'hybrid', 'crossattn'):
            raise ValueError('Unknown adapter type')
        if not 0 <= soft_lambda <= 1 or len(widths) != 4 or min(widths) < 1:
            raise ValueError('Invalid soft_lambda/adapter widths')
        if adapter_type == 'selfattn' and shared:
            raise ValueError('selfattn band-specific stem requires shared=False')
        if adapter_type == 'hybrid' and shared:
            raise ValueError('hybrid band-specific stem requires shared=False')
        self.config = dict(spec=spec, transform=transform, reliability=reliability, gate=gate, support=support,
                           widths=list(widths), gate_max=gate_max, gate_init=gate_init, timesteps=timesteps,
                           soft_lambda=soft_lambda, shared=shared, coarse_only=coarse_only, adapter_type=adapter_type,
                           cross_attention_dim=cross_attention_dim, cross_attention_heads=cross_attention_heads)
        self.spec = spec
        # Non-gradient RMS buffers; EMA updates occur only after successful optimizer steps.
        self.register_buffer('band_rms', torch.ones(10))
        self.register_buffer('rms_fitted', torch.tensor(False))
        self.register_buffer('rms_second_moment', torch.ones(10))
        self.register_buffer('rms_updates', torch.tensor(0, dtype=torch.long))
        # Keep q channels even in no-q controls, set them to zero. Parameter count stays identical B2-B7.
        in_channels = [192,48,12,4] if transform != 'rgb' else [192,192,192,192]
        def make_adapter(c):
            if adapter_type == 'legacy':
                return Adapter(c, widths)
            if adapter_type == 'unet':
                return UNetAdapter(c, widths)
            if adapter_type == 'selfattn':
                return SelfAttnUNetAdapter(c, widths, attention_head_dim=8)
            if adapter_type == 'hybrid':
                return HybridSelfAttnUNetAdapter(c, widths, attention_head_dim=8)
            return CrossAttnUNetAdapter(c, widths, cross_attention_dim, cross_attention_heads)

        if shared:
            self.projections = nn.ModuleList(nn.Conv2d(c, widths[0], 1) for c in in_channels)
            self.adapters = nn.ModuleList([make_adapter(widths[0])])
        else:
            self.projections = nn.ModuleList()
            self.adapters = nn.ModuleList(make_adapter(c) for c in in_channels)
        self.gates = Gates(gate, gate_max, gate_init, timesteps)
        self.zero = nn.ModuleList(nn.Conv2d(4*widths[s['scale']], s['channels'], 1, bias=False) for s in spec['slots'])
        for layer in self.zero:
            nn.init.zeros_(layer.weight)
        # Runtime interventions contain no learned parameters, but are persisted in config.json
        # so ablation/training settings remain inspectable and round-trip through checkpoints.
        self.set_interventions(
            drop_bands=drop_bands,
            drop_scales=drop_scales,
            drop_interval=drop_interval,
            gate_override=gate_override,
            wave_strength=wave_strength,
        )

    def set_interventions(self, drop_bands=None, drop_scales=None, drop_interval=None,
                          gate_override=None, wave_strength=1.):
        """Configure band/scale/time interventions and persist a JSON-safe canonical form.

        `drop_bands` is stored by band name in config but converted to integer indices at runtime.
        `drop_scales` is sorted/canonicalized because order has no semantic meaning.
        `gate_override` is stored as a scalar (not a Tensor) so config.json remains serializable.
        """
        drop_bands = [] if drop_bands is None else list(drop_bands)
        drop_scales = [] if drop_scales is None else list(drop_scales)

        unknown_bands = [b for b in drop_bands if b not in BANDS]
        if unknown_bands:
            raise ValueError(f'Unknown drop bands: {unknown_bands}')
        # Stable ordering prevents false config mismatches when equivalent CLI orders differ.
        band_set = set(drop_bands)
        drop_bands = [name for name in BANDS if name in band_set]

        try:
            drop_scales = [int(s) for s in drop_scales]
        except (TypeError, ValueError) as exc:
            raise ValueError('drop_scales must contain integers') from exc
        invalid_scales = [s for s in drop_scales if s not in range(4)]
        if invalid_scales:
            raise ValueError(f'Invalid drop scales: {invalid_scales}')
        drop_scales = sorted(set(drop_scales))

        if drop_interval is not None:
            if len(drop_interval) != 2:
                raise ValueError('drop_interval must contain [lo, hi]')
            lo, hi = map(float, drop_interval)
            if not 0.0 <= lo <= hi <= 1.0:
                raise ValueError('drop_interval requires 0 <= lo <= hi <= 1')
            drop_interval = [lo, hi]

        if gate_override is not None:
            gate_override = float(gate_override)
            if not math.isfinite(gate_override):
                raise ValueError('gate_override must be finite')

        wave_strength = float(wave_strength)
        if not math.isfinite(wave_strength) or wave_strength < 0:
            raise ValueError('wave_strength must be finite and >= 0')

        self.drop_bands = {BANDS.index(name) for name in drop_bands}
        self.drop_scales = set(drop_scales)
        self.drop_interval = None if drop_interval is None else tuple(drop_interval)
        self.gate_override = gate_override
        self.strength = wave_strength

        self.config.update({
            'drop_bands': drop_bands,
            'drop_scales': drop_scales,
            'drop_interval': drop_interval,
            'gate_override': gate_override,
            'wave_strength': wave_strength,
        })

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

    def encode(self, image, hole, encoder_hidden_states=None):
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
        if cfg['adapter_type'] == 'crossattn':
            if encoder_hidden_states is None:
                raise ValueError('crossattn adapter requires encoder_hidden_states')
            if cfg['shared']:
                features = [self.adapters[0](proj(x), encoder_hidden_states) for proj,x in zip(self.projections,inputs)]
            else:
                features = [adapter(x, encoder_hidden_states) for adapter,x in zip(self.adapters,inputs)]
        elif cfg['shared']:
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
            g = torch.full_like(g, self.gate_override)
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

    def forward(self, image, hole, timesteps, encoder_hidden_states=None):
        return self.project(self.encode(image,hole,encoder_hidden_states),timesteps)

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
