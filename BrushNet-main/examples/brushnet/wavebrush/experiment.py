"""Common generation/evaluation logic. Accelerator assigns one shard per process."""
import argparse
import gc
import inspect
import json
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from .core import WaveConditioner
from .data import load_records, load_pair, sample_name, sample_seed
from .integration import wave_inference


def parser(evaluate=False):
    p = argparse.ArgumentParser()
    p.add_argument('--base_model_path', default='data/ckpt/realisticVisionV60B1_v51VAE')
    p.add_argument('--brushnet_ckpt_path', default='data/ckpt/segmentation_mask_brushnet_ckpt')
    p.add_argument('--wave_path', help='Saved wave folder; omit only with --baseline')
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--mapping_file', help='Original BrushBench JSON or JSONL manifest')
    p.add_argument('--base_dir', default='.')
    p.add_argument('--image', help='Single image, alternative to mapping_file')
    p.add_argument('--mask', help='Single binary mask, WHITE=hole')
    p.add_argument('--prompt', default='')
    p.add_argument('--mask_key', default='inpainting_mask')
    p.add_argument('--image_save_path', '--output', dest='output', required=True)
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_inference_steps', type=int, default=50)
    p.add_argument('--guidance_scale', type=float, default=7.5)
    p.add_argument('--paintingnet_conditioning_scale', '--brushnet_conditioning_scale', dest='scale', type=float, default=1.)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--seed_policy', choices=['per_id', 'same'], default='per_id')
    p.add_argument('--blended', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--generate', action='store_true', default=not evaluate)
    p.add_argument('--metrics', choices=['none','reference','light','full'], default='full' if evaluate else 'none')
    p.add_argument('--metric_ckpt_path', default='data/ckpt')
    p.add_argument('--precision', choices=['fp32','fp16','bf16'], default='bf16')
    p.add_argument('--drop_bands', nargs='*', choices=['H1','H2','H3','L3'], default=[])
    p.add_argument('--drop_scales', type=int, nargs='*', choices=[0,1,2,3], default=[])
    p.add_argument('--drop_interval', type=float, nargs=2)
    p.add_argument('--gate_override', type=float, help='Fixed scalar at inference, for intervention only')
    p.add_argument('--wave_strength', type=float, default=1.)
    return p


def write_jsonl(path, rows):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    tmp.replace(path)


def merged_rows(output, prefix, world, expected):
    rows = []
    for rank in range(world):
        path = output / f'{prefix}.rank{rank}.jsonl'
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise RuntimeError(f'{prefix}: missing or duplicate sample ids')
    order = {key: i for i,key in enumerate(expected)}
    return sorted(rows, key=lambda row: order[row['id']])


def blend(pred, gt, mask):
    hole = np.array(mask, dtype=np.float32)/255
    blurred = cv2.GaussianBlur(hole, (21,21), 0)
    alpha = 1-(1-hole)*(1-blurred)
    pixels = np.asarray(gt)*(1-alpha[:,:,None]) + np.asarray(pred)*alpha[:,:,None]
    return Image.fromarray(np.clip(pixels,0,255).astype(np.uint8))


def main(evaluate=False):
    a = parser(evaluate).parse_args()
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    acc = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))])
    if a.resolution % 64 or a.batch_size < 1:
        raise ValueError('resolution must be divisible by 64, batch_size must be positive')
    if a.drop_interval and not 0 <= a.drop_interval[0] <= a.drop_interval[1] <= 1:
        raise ValueError('drop_interval must satisfy 0 <= lo <= hi <= 1')
    if a.generate and bool(a.wave_path) == bool(a.baseline):
        raise ValueError('Choose exactly one of --wave_path and --baseline for generation')
    if a.mapping_file:
        records = load_records(a.mapping_file)
    elif a.image and a.mask:
        records = [dict(id='single', image=str(Path(a.image).resolve()),
                        mask=str(Path(a.mask).resolve()), caption=a.prompt)]
    else:
        raise ValueError('Provide --mapping_file or both --image and --mask')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    local = records[acc.process_index::acc.num_processes]
    expected = [r['id'] for r in records]
    dtype = {'fp32':torch.float32, 'fp16':torch.float16, 'bf16':torch.bfloat16}[a.precision]
    if acc.device.type == 'cpu':
        dtype = torch.float32
    if a.generate:
        # Check all inputs/settings before honoring existing outputs.
        settings = {k:v for k,v in vars(a).items() if k not in ('metrics','metric_ckpt_path','resume','generate')}
        settings.update(records=records, world_size=acc.num_processes)
        config_path = output / 'generation_config.json'
        if acc.is_main_process:
            if config_path.exists():
                if not a.resume or json.loads(config_path.read_text()) != settings:
                    raise ValueError('Output already has a run; use NEW output or --resume with identical settings')
            else:
                if a.resume:
                    raise ValueError('Cannot resume without generation_config.json')
                config_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False))
            for subdir in ('raw','selected'):
                (output/subdir).mkdir(exist_ok=True)
        acc.wait_for_everyone()
        from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
        base = BrushNetModel.from_pretrained(a.brushnet_ckpt_path, torch_dtype=dtype)
        pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            a.base_model_path, brushnet=base, torch_dtype=dtype, low_cpu_mem_usage=False,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(acc.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=not acc.is_main_process)
        wave = WaveConditioner.from_pretrained(a.wave_path, acc.device).to(dtype=dtype).eval() if a.wave_path else None
        if wave is not None:
            if wave.config['timesteps'] != pipe.scheduler.config.num_train_timesteps:
                raise ValueError('Wave and diffusion scheduler training timestep count differs')
            wave.drop_bands = {['H1','H2','H3','L3'].index(b) for b in a.drop_bands}
            wave.drop_scales = set(a.drop_scales)
            wave.drop_interval = a.drop_interval
            wave.strength = a.wave_strength
            if a.gate_override is not None:
                wave.gate_override = torch.full((4,4), a.gate_override, device=acc.device)
        parameters = inspect.signature(pipe.__call__).parameters
        scale_name = 'brushnet_conditioning_scale' if 'brushnet_conditioning_scale' in parameters else 'paintingnet_conditioning_scale'
        if scale_name not in parameters:
            raise ValueError('Custom pipeline has neither brushnet_conditioning_scale nor paintingnet_conditioning_scale')
        generated = []
        for start in range(0,len(local),a.batch_size):
            subset = local[start:start+a.batch_size]
            pending = [r for r in subset if not (a.resume and (output/'raw'/sample_name(r['id'])).exists()
                                                 and (output/'selected'/sample_name(r['id'])).exists())]
            if pending:
                pairs = [load_pair(r,a.base_dir,a.mask_key,a.resolution) for r in pending]
                images = [Image.composite(Image.new('RGB',gt.size),gt,mask) for gt,mask in pairs]
                masks = [mask for _,mask in pairs]
                generators = [torch.Generator(device=acc.device).manual_seed(
                    a.seed if a.seed_policy == 'same' else sample_seed(a.seed,r['id'])) for r in pending]
                with torch.inference_mode(), wave_inference(base,wave,images,masks):
                    results = pipe([r.get('caption','') for r in pending], images, masks,
                                   num_inference_steps=a.num_inference_steps, generator=generators,
                                   guidance_scale=a.guidance_scale, **{scale_name:a.scale}).images
                if acc.is_main_process and hasattr(pipe.scheduler, 'timesteps'):
                    (output/'sampler_timesteps.json').write_text(json.dumps(torch.as_tensor(pipe.scheduler.timesteps).cpu().tolist()))
                for row,pred,(gt,mask) in zip(pending,results,pairs):
                    name = sample_name(row['id'])
                    pred.save(output/'raw'/name)
                    (blend(pred,gt,mask) if a.blended else pred).save(output/'selected'/name)
            generated += [dict(id=r['id'],raw='raw/'+sample_name(r['id']),
                               selected='selected/'+sample_name(r['id']),image=r['image'],
                               seed=a.seed if a.seed_policy=='same' else sample_seed(a.seed,r['id'])) for r in subset]
        write_jsonl(output/f'results.rank{acc.process_index}.jsonl',generated)
        del pipe, base, wave
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        acc.wait_for_everyone()
        if acc.is_main_process:
            write_jsonl(output/'results.jsonl', merged_rows(output,'results',acc.num_processes,expected))
        acc.wait_for_everyone()
    if a.metrics == 'none':
        return
    from .metrics import reference_metrics
    index_path = output/'results.jsonl'
    if not index_path.exists():
        raise ValueError('Missing results.jsonl: run generation with --generate first')
    saved_settings = json.loads((output/'generation_config.json').read_text())
    if saved_settings['records'] != records or saved_settings['resolution'] != a.resolution or saved_settings['mask_key'] != a.mask_key or saved_settings['base_dir'] != a.base_dir:
        raise ValueError('Evaluation manifest/preprocessing differs from generation')
    index_rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    index = {row['id']:row for row in index_rows}
    if len(index) != len(index_rows) or set(index) != set(expected):
        raise ValueError('Evaluation manifest differs from generated ids')
    evaluator = None
    if a.metrics in ('light','full'):
        from .local_evaluator import BrushNetValidationEvaluator
        evaluator = BrushNetValidationEvaluator(acc.device, ckpt_path=a.metric_ckpt_path, offload=True)
    evaluated = []
    try:
        for row in local:
            gt,mask = load_pair(row,a.base_dir,a.mask_key,a.resolution)
            result = dict(id=row['id'],image=row['image'],seed=index[row['id']]['seed'],variant='both')
            for version in ('raw','selected'):
                pred = Image.open(output/index[row['id']][version]).convert('RGB')
                if pred.size != gt.size:
                    raise ValueError('Prediction dimensions differ; evaluation never silently resizes outputs')
                values = reference_metrics(np.asarray(pred)/255.,np.asarray(gt)/255.,np.asarray(mask)/255.)
                result.update({version+'_'+k: v for k,v in values.items()})
                if evaluator is not None:
                    scores = evaluator.evaluate(gt,pred,mask,prompt=row.get('caption',''),full=a.metrics=='full')
                    result.update({version+'_'+k: v for k,v in scores.items()})
            evaluated.append(result)
    finally:
        if evaluator is not None:
            evaluator.close()
    write_jsonl(output/f'metrics.rank{acc.process_index}.jsonl',evaluated)
    acc.wait_for_everyone()
    if acc.is_main_process:
        import csv
        rows = merged_rows(output,'metrics',acc.num_processes,expected)
        columns = list(rows[0]) if rows else ['id','image']
        with (output/'evaluation_result.csv').open('w',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        summary = {}
        for key in columns:
            if key in ('id','image','seed','variant'):
                continue
            all_values = [r[key] for r in rows]
            finite = [v for v in all_values if math.isfinite(v)]
            summary[key] = dict(mean=float(np.mean(finite)) if finite else None,
                                finite_count=len(finite), total=len(rows),
                                nan_count=sum(math.isnan(v) for v in all_values),
                                inf_count=sum(math.isinf(v) for v in all_values))
        (output/'evaluation_summary.json').write_text(json.dumps(summary,indent=2))
        with (output/'evaluation_result_sum.csv').open('w',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=['metric','mean','finite_count','total','nan_count','inf_count'])
            writer.writeheader()
            writer.writerows(dict(metric=k,**v) for k,v in summary.items())
        print(f'Evaluated {len(rows)} samples: {output}')
    acc.wait_for_everyone()
