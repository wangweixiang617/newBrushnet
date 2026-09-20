"""BrushNet residual integration; never patch the installed diffusers package."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from .core import (
    WaveConditioner, PRESETS, slot_spec, flatten_residuals, merge_residuals,
    haar_pyramid, BANDS,
)


def add_wave_args(parser):
    parser.add_argument('--wave_preset', choices=['B0']+list(PRESETS), default='recommended')
    parser.add_argument('--wave_rms', type=str)  #rms的路径
    parser.add_argument('--wave_resume', type=str, help='Warm-start wave folder; not optimizer resume')
    parser.add_argument('--wave_widths', type=int, nargs=4, default=[32,64,96,128])
    parser.add_argument('--wave_adapter', choices=['legacy','unet','selfattn','hybrid','crossattn'], default='unet')
    parser.add_argument('--wave_gate', choices=['fixed','constant','time'])
    parser.add_argument('--wave_gate_max', type=float, default=2.)
    parser.add_argument('--wave_gate_init', type=float, default=1.)
    parser.add_argument('--wave_support', choices=['directional','average','hard'], default='directional')
    parser.add_argument('--wave_reliability', choices=['none','concat','premul','soft'])
    parser.add_argument('--wave_shared', action='store_true')
    parser.add_argument('--wave_coarse_only', action='store_true')
    parser.add_argument('--wave_soft_lambda', type=float, default=.5)
    parser.add_argument('--train_brushnet', action='store_true', help='Joint fine-tuning (default in the minimal training entrypoint)')
    parser.add_argument('--disable_validation', action='store_true')
    parser.add_argument('--drop_bands', nargs='*', choices=['H1', 'H2', 'H3', 'L3'], default=[])
    parser.add_argument('--drop_scales', nargs='*', type=int, choices=[0, 1, 2, 3], default=[])
    parser.add_argument('--drop_interval', nargs=2, type=float)
    parser.add_argument('--gate_override', type=float)
    parser.add_argument('--wave_strength', type=float, default=1.)
    parser.add_argument('--wave_log_every', type=int, default=10)


@torch.no_grad()
def build_wave(brushnet, args, device, timesteps=1000):
    if args.wave_preset == 'B0':
        return None
    # Probe the actual fork rather than assuming down/mid/up slot counts or channels.
    brushnet.to(device)
    mode = brushnet.training
    brushnet.eval()
    p = next(brushnet.parameters())
    res = args.resolution
    if res % 64:
        raise ValueError('resolution must be divisible by 64')
    cross_dim = brushnet.config.cross_attention_dim
    if not isinstance(cross_dim,int):
        raise ValueError('This implementation targets SD1.5 BrushNet, not SDXL')
    result = brushnet(torch.zeros(1,4,res//8,res//8,device=device,dtype=p.dtype),
                      torch.tensor([500],device=device),
                      encoder_hidden_states=torch.zeros(1,77,cross_dim,device=device,dtype=p.dtype),
                      brushnet_cond=torch.zeros(1,5,res//8,res//8,device=device,dtype=p.dtype),return_dict=False)
    spec = slot_spec(result,(res//8,res//8))
    brushnet.train(mode)
    if args.wave_resume:
        wave = WaveConditioner.from_pretrained(args.wave_resume,device)
        if wave.spec != spec:
            raise ValueError('Saved wave residual schema differs from current BrushNet')
    else:
        options = dict(PRESETS[args.wave_preset])
        if args.wave_gate:
            options['gate'] = args.wave_gate
        if args.wave_reliability:
            options['reliability'] = args.wave_reliability
        wave = WaveConditioner(spec,**options,widths=args.wave_widths,support=args.wave_support,
                               gate_max=args.wave_gate_max,gate_init=args.wave_gate_init,timesteps=timesteps,
                               soft_lambda=args.wave_soft_lambda,shared=args.wave_shared,coarse_only=args.wave_coarse_only,
                               adapter_type=args.wave_adapter,cross_attention_dim=cross_dim,cross_attention_heads=8).to(device)
        if options['transform'] != 'rgb' and args.wave_rms:
            data = json.loads(Path(args.wave_rms).read_text())
            if data.get('resolution') != args.resolution:
                raise ValueError('RMS resolution differs from training resolution')
            wave.set_rms(args.wave_rms)

    # Apply current-run interventions for both fresh construction and --wave_resume.
    # set_interventions() also stores a canonical JSON-safe copy in wave.config.
    wave.set_interventions(
        drop_bands=args.drop_bands,
        drop_scales=args.drop_scales,
        drop_interval=args.drop_interval,
        gate_override=args.gate_override,
        wave_strength=args.wave_strength,
    )
    return wave


def image_tensors(images, masks, device, size=None):
    """PIL images may already be black-masked; masks white=hole. No VAE preprocessing reuse."""
    xs, ms = [], []
    for im, ma in zip(images,masks):
        im,ma = im.convert('RGB'),ma.convert('L')
        if size:
            im=im.resize(size,Image.Resampling.BICUBIC)
            ma=ma.resize(size,Image.Resampling.NEAREST)
        if im.size != ma.size:
            raise ValueError('Image and mask sizes differ')
        x=torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()/127.5-1
        m=torch.from_numpy((np.array(ma,copy=True)>127).astype('float32'))[None]
        xs.append(x); ms.append(m)
    return torch.stack(xs).to(device),torch.stack(ms).to(device)


@contextmanager
def wave_inference(brushnet,wave,images,masks,trace=None):
    """One pipeline call, num_images_per_prompt=1; standard CFG [uncond batch, cond batch].
    Single BrushNet only. guess_mode/global_pool_conditions are deliberately rejected.
    Restores hook and training mode even when inference raises an exception.
    """
    if wave is None:
        yield
        return
    if torch.is_grad_enabled():
        # Inference uses a detached cache; training must call wave(image, hole, t) normally.
        pass
    device=next(wave.parameters()).device
    old_mode=wave.training
    wave.eval()
    if len(images) != len(masks):
        raise ValueError('Image/mask batch differs')
    x,m=image_tensors(images,masks,device)
    input_stats = None
    if trace is not None:
        with torch.no_grad():
            input_stats = wave_input_stats(wave, x, m)
    crossattn = wave.config.get('adapter_type') == 'crossattn'
    features = None
    if not crossattn:
        with torch.no_grad():
            features=wave.encode(x,m)
    cross_cache = {}
    batch=len(images)
    calls=[0]

    def hook(module,args,kwargs,result):
        if kwargs.get('guess_mode',False) or getattr(module.config,'global_pool_conditions',False):
            raise ValueError('wave_inference supports guess_mode=False and global_pool_conditions=False')
        sample=args[0] if args else kwargs['sample']
        t=args[1] if len(args)>1 else kwargs['timestep']
        n=sample.shape[0]
        if n not in (batch,2*batch):
            raise ValueError('Only num_images_per_prompt=1 and ordinary CFG are supported')
        if crossattn:
            encoder_hidden_states = kwargs.get('encoder_hidden_states')
            if encoder_hidden_states is None:
                raise ValueError('crossattn wave adapter requires BrushNet encoder_hidden_states')
            if n not in cross_cache:
                xi = x if n == batch else x.repeat(2,1,1,1)
                mi = m if n == batch else m.repeat(2,1,1,1)
                with torch.no_grad():
                    cross_cache[n] = wave.encode(xi,mi,encoder_hidden_states)
            cached = cross_cache[n]
        else:
            cached=features if n==batch else [[f.repeat(2,1,1,1) for f in branch] for branch in features]
        with torch.no_grad():
            extra=wave.project(cached,t)
        if trace is not None:
            base,_,_=flatten_residuals(result)
            #换个精度试试 临时关闭bf16
            with torch.autocast(device_type=device.type,enabled=False):
                gate_tensor=wave.effective_gates(torch.as_tensor(t,device=device).reshape(-1)).detach().float().cpu()
            gates = gate_tensor.numpy().round(8).tolist()

            residual_rows = residual_stats(base, extra)
            projected_branch_rms = wave.branch_rms(cached, t)
            trace_row = dict(
                call=calls[0],
                t=float(torch.as_tensor(t).flatten()[0]),
                gates=gates,
                gate_summary=gate_summary(gate_tensor),
                residuals=residual_rows,
                residual_summary=residual_summary(residual_rows),
                branch_rms=projected_branch_rms,
                branch_summary=branch_summary(projected_branch_rms),
            )
            # Raw MVWT/q and cached adapter features do not change across diffusion
            # timesteps in the ordinary cached inference path, so store them once.
            if calls[0] == 0:
                trace_row['wave_input'] = input_stats
                trace_row['feature_stats'] = feature_rms_stats(cached)
            trace.append(trace_row)
        calls[0]+=1
        merged=merge_residuals(result,extra)
        if isinstance(result,(tuple,list)):
            return merged
        return type(result)(down_block_res_samples=merged[0],mid_block_res_sample=merged[1],up_block_res_samples=merged[2])
    handle=brushnet.register_forward_hook(hook,with_kwargs=True)
    try:
        yield
        if calls[0]==0:
            raise RuntimeError('BrushNet hook was never called: check pipeline component identity')
    finally:
        handle.remove()
        wave.train(old_mode)


@torch.no_grad()
def residual_stats(base,extra):
    rows=[]
    for i,(b,w) in enumerate(zip(base,extra)):
        bf,wf=b.detach().float(),w.detach().float()
        br,wr=bf.square().mean().sqrt().item(),wf.square().mean().sqrt().item()
        cross = (bf * wf).mean().item()
        cosine = torch.nn.functional.cosine_similarity(
            bf.reshape(1, -1),
            wf.reshape(1, -1),
        ).item()
        merged = bf + wf
        merged_rms = merged.square().mean().sqrt().item()
        rows.append(dict(
            slot=i,
            shape=list(b.shape),
            brush_rms=br,
            wave_rms=wr,
            wave_mean=wf.mean().item(),
            wave_abs_max=wf.abs().max().item(),
            ratio=wr/max(br,1e-8),
            cross_mean=cross,
            cosine=cosine,
            merged_rms=merged_rms,
            merged_over_brush=merged_rms/max(br,1e-8),
        ))
    return rows


@torch.no_grad()
def residual_summary(rows):
    """Compact residual diagnostics for one diffusion timestep."""
    if not rows:
        return {}
    ratios = torch.tensor([r['ratio'] for r in rows], dtype=torch.float32)
    cosines = torch.tensor([r['cosine'] for r in rows], dtype=torch.float32)
    merged_ratios = torch.tensor([r['merged_over_brush'] for r in rows], dtype=torch.float32)
    return {
        'ratio_mean': float(ratios.mean()),
        'ratio_median': float(ratios.median()),
        'ratio_p90': float(torch.quantile(ratios, 0.90)),
        'ratio_p95': float(torch.quantile(ratios, 0.95)),
        'ratio_max': float(ratios.max()),
        'ratio_gt_025': float((ratios > 0.25).float().mean()),
        'ratio_gt_050': float((ratios > 0.50).float().mean()),
        'ratio_gt_100': float((ratios > 1.00).float().mean()),
        'cosine_mean': float(cosines.mean()),
        'cosine_abs_mean': float(cosines.abs().mean()),
        'cosine_positive_fraction': float((cosines > 0).float().mean()),
        'merged_over_brush_mean': float(merged_ratios.mean()),
        'merged_over_brush_max': float(merged_ratios.max()),
    }


@torch.no_grad()
def wave_input_stats(wave, image, hole):
    """Log raw/normalized wave bands and reliability q without changing training."""
    cfg = wave.config
    result = {
        'transform': cfg['transform'],
        'reliability': cfg['reliability'],
        'support': cfg['support'],
        'drop_bands': cfg.get('drop_bands', []),
        'drop_scales': cfg.get('drop_scales', []),
        'drop_interval': cfg.get('drop_interval'),
        'gate_override': cfg.get('gate_override'),
        'wave_strength': cfg.get('wave_strength', 1.0),
        'rms_fitted': bool(wave.rms_fitted.item()),
        'rms_updates': int(wave.rms_updates.item()),
        'band_rms_buffer': [float(v) for v in wave.band_rms.detach().float().cpu()],
    }

    if cfg['transform'] == 'rgb':
        x = image.detach().float() * (1.0 - hole.detach().float())
        result['rgb'] = {
            'rms': float(x.square().mean().sqrt()),
            'mean': float(x.mean()),
            'abs_max': float(x.abs().max()),
        }
        return result

    bands, qs = haar_pyramid(
        image,
        hole,
        transform=cfg['transform'],
        support=cfg['support'],
    )
    band_stats = {}
    q_stats = {}

    for i, (name, w, q) in enumerate(zip(BANDS, bands, qs)):
        wf = w.detach().float()
        qf = q.detach().float()
        bstat = {
            'shape': list(w.shape),
            'raw_rms': float(wf.square().mean().sqrt()),
            'raw_mean': float(wf.mean()),
            'raw_abs_max': float(wf.abs().max()),
        }

        if i < 3:
            bstat['direction_rms'] = {
                'LH': float(wf[:, 0:3].square().mean().sqrt()),
                'HL': float(wf[:, 3:6].square().mean().sqrt()),
                'HH': float(wf[:, 6:9].square().mean().sqrt()),
            }
            rms = wave.band_rms[i*3:i*3+3].detach().float()
        else:
            rms = wave.band_rms[9:10].detach().float()

        denom = rms.repeat_interleave(3)[None, :, None, None].clamp_min(1e-6)
        wn = wf / denom
        bstat['normalized_rms'] = float(wn.square().mean().sqrt())
        band_stats[name] = bstat

        qstat = {
            'shape': list(q.shape),
            'mean': float(qf.mean()),
            'std': float(qf.std(unbiased=False)),
            'min': float(qf.min()),
            'max': float(qf.max()),
            'lt_025_fraction': float((qf < 0.25).float().mean()),
            'lt_050_fraction': float((qf < 0.50).float().mean()),
            'gt_075_fraction': float((qf > 0.75).float().mean()),
            'gt_090_fraction': float((qf > 0.90).float().mean()),
        }
        if i < 3 and qf.shape[1] == 3:
            qstat['direction_mean'] = {
                'LH': float(qf[:, 0].mean()),
                'HL': float(qf[:, 1].mean()),
                'HH': float(qf[:, 2].mean()),
            }
        q_stats[name] = qstat

    result['bands'] = band_stats
    result['reliability_q'] = q_stats
    return result


@torch.no_grad()
def feature_rms_stats(features):
    """Adapter cached feature statistics: four bands x four scales."""
    result = {}
    for b, name in enumerate(BANDS):
        result[name] = {}
        for s, f in enumerate(features[b]):
            ff = f.detach().float()
            result[name][f'scale{s}'] = {
                'shape': list(f.shape),
                'rms': float(ff.square().mean().sqrt()),
                'mean': float(ff.mean()),
                'abs_max': float(ff.abs().max()),
            }
    return result


@torch.no_grad()
def branch_summary(branch_values):
    """Summarize projected per-band RMS. normalized_proxy is descriptive, not additive energy."""
    if not branch_values:
        return {}
    x = torch.tensor(branch_values, dtype=torch.float32)
    result = {
        'slot_mean_rms': {
            name: float(x[:, b].mean()) for b, name in enumerate(BANDS)
        },
        'slot_max_rms': {
            name: float(x[:, b].max()) for b, name in enumerate(BANDS)
        },
    }
    proxy = x.mean(dim=0)
    total = proxy.sum()
    if float(total) > 0:
        proxy = proxy / total
        result['normalized_proxy'] = {
            name: float(proxy[b]) for b, name in enumerate(BANDS)
        }
    return result


@torch.no_grad()
def gate_summary(g):
    """Compact band/scale gate statistics for one diffusion timestep."""
    gf = g.detach().float()
    return {
        'mean': float(gf.mean()),
        'min': float(gf.min()),
        'max': float(gf.max()),
        'band_mean': {
            BANDS[b]: float(gf[:, b, :].mean()) for b in range(4)
        },
        'scale_mean': {
            f'scale{s}': float(gf[:, :, s].mean()) for s in range(4)
        },
    }


def append_jsonl(path,row):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')

def conditioning_scale_kwargs(pipe, value):
    # Official and historical local pipeline signatures use different parameter names.
    import inspect
    parameters = inspect.signature(pipe.__call__).parameters
    for name in ('brushnet_conditioning_scale', 'paintingnet_conditioning_scale'):
        if name in parameters:
            return {name: float(value)}
    raise ValueError('Pipeline has no supported BrushNet conditioning-scale argument')
