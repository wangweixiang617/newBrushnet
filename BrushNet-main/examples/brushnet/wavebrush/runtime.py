"""Checkpoint and validation helpers shared by the patched training entrypoint."""
import contextlib
import copy
import functools
import json
import random
from pathlib import Path
import numpy as np
import torch
from .core import WaveConditioner


def register_model_hooks(accelerator):
    def save(models, weights, output_dir):
        for model in models:
            model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                model.save_pretrained(Path(output_dir) / ('wave' if isinstance(model, WaveConditioner) else 'brushnet'))
        weights.clear()

    def load(models, input_dir):
        while models:
            model = accelerator.unwrap_model(models.pop())
            if isinstance(model, WaveConditioner):
                loaded = WaveConditioner.from_pretrained(Path(input_dir) / 'wave')
                if model.config != loaded.config:
                    raise ValueError('Wave config changed on resume; use --wave_resume only for a warm start')
            else:
                loaded = type(model).from_pretrained(Path(input_dir) / 'brushnet')
            model.load_state_dict(loaded.state_dict())
    accelerator.register_save_state_pre_hook(save)
    accelerator.register_load_state_pre_hook(load)


def protocol(args, accelerator, dataset):
    keys = ('resolution', 'train_batch_size', 'gradient_accumulation_steps', 'seed', 'random_mask',
            'proportion_empty_prompts', 'train_brushnet', 'wave_preset', 'rms_mode', 'rms_ema_decay',
            'learning_rate', 'lr_scheduler', 'lr_warmup_steps', 'max_train_steps', 'mixed_precision',
            'pretrained_model_name_or_path', 'brushnet_model_name_or_path')
    return dict(world_size=accelerator.num_processes, dataset=dataset.fingerprint, samples=len(dataset),
                options={k: getattr(args, k) for k in keys})


def save_progress(path, state):
    path = Path(path) / 'wave_progress.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def read_progress(path, expected):
    marker = Path(path) / 'wave_progress.json'
    if not marker.exists():
        raise ValueError('Incomplete or old checkpoint: missing wave_progress.json; warm-start weights instead')
    state = json.loads(marker.read_text())
    if state['protocol'] != expected:
        raise ValueError('Resume data/world size/batch/training settings changed. Use unchanged settings or warm-start.')
    return state


def validation_guard_old(fn):
    """Protect training RNG/modes; validation never enters a DDP forward or updates RMS."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        import inspect
        values = inspect.signature(fn).bind(*args, **kwargs)
        values.apply_defaults()
        a = values.arguments
        accelerator = a['accelerator']
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda = torch.cuda.get_rng_state(accelerator.device) if accelerator.device.type == 'cuda' else None
        models = [a.get(k) for k in ('vae', 'unet', 'text_encoder', 'brushnet')]
        models = [accelerator.unwrap_model(m) for m in models if m is not None]
        saved = [(m, m.training, next(m.parameters()).dtype) for m in models]
        try:
            if a.get('wave') is not None:
                raw = accelerator.unwrap_model(a['wave'])
                a['wave'] = WaveConditioner(**raw.config).to(accelerator.device)
                a['wave'].load_state_dict(raw.state_dict())
                a['wave'].eval()
            if a.get('brushnet') is not None:
                a['brushnet'] = accelerator.unwrap_model(a['brushnet'])
            for model, _, _ in saved:
                model.eval()
            return fn(**a)
        finally:
            for model, mode, dtype in saved:
                model.to(dtype=dtype)
                model.train(mode)
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state(cuda, accelerator.device)
    return wrapped

def validation_guard(fn):
    """
    Protect training state during validation.

    保留：
    1. Python / NumPy / Torch / CUDA RNG 状态
    2. unwrap Accelerator/DDP 模型
    3. validation 前切换到 eval()
    4. validation 后恢复原 train/eval 状态
    5. validation 后恢复 RNG 状态

    不再：
    1. clone / deepcopy WaveConditioner
    2. 重新 load_state_dict
    3. 对模型执行多余的 dtype .to()
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        import inspect

        # -------------------------------------------------
        # 1. 绑定原函数参数
        # -------------------------------------------------
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments

        accelerator = a["accelerator"]

        # -------------------------------------------------
        # 2. 保存 RNG 状态
        # -------------------------------------------------
        py_rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()

        cuda_rng_state = None
        if accelerator.device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(accelerator.device)

        # -------------------------------------------------
        # 3. unwrap 模型，并记录原 train/eval 状态
        # -------------------------------------------------
        saved_modes = []

        for name in (
            "vae",
            "unet",
            "text_encoder",
            "brushnet",
            "wave",
        ):
            model = a.get(name)

            if model is None:
                continue

            # DDP / Accelerator wrapper -> 原始 module
            model = accelerator.unwrap_model(model)

            # validation 函数内部使用 unwrap 后的模型
            a[name] = model

            # 保存原始 mode
            saved_modes.append(
                (model, model.training)
            )

            # validation 统一使用 eval
            model.eval()

        try:
            # -------------------------------------------------
            # 4. 执行 validation
            # -------------------------------------------------
            return fn(*bound.args, **bound.kwargs)

        finally:
            # -------------------------------------------------
            # 5. 恢复模型原来的 train/eval 状态
            # -------------------------------------------------
            for model, was_training in saved_modes:
                model.train(was_training)

            # -------------------------------------------------
            # 6. 恢复 RNG
            # -------------------------------------------------
            random.setstate(py_rng_state)
            np.random.set_state(np_rng_state)
            torch.set_rng_state(torch_rng_state)

            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(
                    cuda_rng_state,
                    accelerator.device,
                )

    return wrapped