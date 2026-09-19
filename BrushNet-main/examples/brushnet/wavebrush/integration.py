"""BrushNet residual integration; never patch the installed diffusers package."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from .core import WaveConditioner, PRESETS, slot_spec, flatten_residuals, merge_residuals


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
        return wave
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
            # gates=wave.effective_gates(torch.as_tensor(t,device=device).reshape(-1)).detach().float().cpu().tolist()
            trace.append(dict(call=calls[0],t=float(torch.as_tensor(t).flatten()[0]),gates=gates,
                              residuals=residual_stats(base,extra),branch_rms=wave.branch_rms(cached,t)))
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
        rows.append(dict(slot=i,shape=list(b.shape),brush_rms=br,wave_rms=wr,wave_mean=wf.mean().item(),
                         wave_abs_max=wf.abs().max().item(),ratio=wr/max(br,1e-8)))
    return rows


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
