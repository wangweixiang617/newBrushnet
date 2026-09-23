from wavebrush.core import WaveConditioner
from wavebrush.integration import wave_inference, conditioning_scale_kwargs
import gc
import argparse
import contextlib
import json
import math
import os
from pathlib import Path

import cv2
import hpsv2
import ImageReward as RM
import numpy as np
import open_clip
import pandas as pd
import torch
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore


# When True, mirror the training-time validation sampling protocol:
# first 24 validation items x 2 stochastic repeats = 48 metric samples.
# Set to False to recover the original full-mapping, single-pass behavior.
ALIGN_TRAIN_VALIDATION_24X2 = True
TRAIN_VALIDATION_COUNT = 24
TRAIN_VALIDATION_REPEATS = 2
# This run was trained/validated in bf16; change only if the training validation precision changes.
TRAIN_VALIDATION_DTYPE = torch.bfloat16


def rle2mask(mask_rle, shape):  # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)


class MetricsCalculator:
    def __init__(self, device, ckpt_path="data/ckpt"):
        self.device = device

        # CLIP / LPIPS
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)

        # Aesthetic model
        self.aesthetic_model = torch.nn.Linear(768, 1)
        aesthetic_model_ckpt_path = os.path.join(ckpt_path, "aesthetic", "sa_0_4_vit_l_14_linear.pth")
        self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path, map_location="cpu"))
        self.aesthetic_model = self.aesthetic_model.to(device).eval()

        # Aesthetic 使用的 OpenAI CLIP
        openclip_path = os.path.join(ckpt_path, "aesthetic", "ViT-L-14.pt")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=openclip_path, load_weights_only=False
        )
        self.clip_model = self.clip_model.to(device).eval()

        # ImageReward 完全使用本地文件
        image_reward_path = os.path.join(ckpt_path, "ImageReward", "ImageReward.pt")
        image_reward_med_config = os.path.join(ckpt_path, "ImageReward", "med_config.json")
        self.imagereward_model = RM.load(image_reward_path, device=device, med_config=image_reward_med_config)

    def calculate_image_reward(self, image, prompt):
        return self.imagereward_model.score(prompt, [image])

    def calculate_hpsv21_score(self, image, prompt):
        return hpsv2.score(image, prompt, hps_version="v2.1")[0].item()

    def calculate_aesthetic_score(self, img):
        image = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            prediction = self.aesthetic_model(image_features)
        return prediction.detach().cpu().item()

    def calculate_clip_similarity(self, img, txt):
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).to(self.device)
        score = self.clip_metric_calculator(img_tensor, txt)
        return score.cpu().item()

    def calculate_psnr(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        difference_square_sum = ((img_pred - img_gt) ** 2).sum()
        difference_size = mask.sum()
        mse = difference_square_sum / difference_size

        if mse < 1.0e-10:
            return 1000
        return 20 * math.log10(1 / math.sqrt(mse))

    def calculate_lpips(self, img_gt, img_pred, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.lpips_metric_calculator(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1)
        return score.cpu().item()

    def calculate_mse(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        difference_square_sum = ((img_pred - img_gt) ** 2).sum()
        difference_size = mask.sum()
        mse = difference_square_sum / difference_size
        return mse.item()


parser = argparse.ArgumentParser()
parser.add_argument("--brushnet_ckpt_path", type=str, default="data/ckpt/segmentation_mask_brushnet_ckpt")
parser.add_argument("--base_model_path", type=str, default="data/ckpt/realisticVisionV60B1_v51VAE")
parser.add_argument("--image_save_path", type=str, default="runs/evaluation_result/BrushBench/brushnet_segmask/inside")
parser.add_argument("--mapping_file", type=str, default="data/BrushBench/mapping_file.json")
parser.add_argument("--base_dir", type=str, default="data/BrushBench")
parser.add_argument("--mask_key", type=str, default="inpainting_mask")
parser.add_argument("--blended", action="store_true")
parser.add_argument("--paintingnet_conditioning_scale", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--wave_path", type=str, default=None, help="Wave folder; default: brushnet_ckpt_path/wave")
parser.add_argument("--disable_wave", action="store_true", help="Evaluate original BrushNet baseline")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument('--guidance_scale', type=float, default=7.5)
parser.add_argument('--metric_ckpt_path', default='data/ckpt')
parser.add_argument('--drop_bands', nargs='*', choices=['H1','H2','H3','L3'], default=[])
parser.add_argument('--drop_scales', nargs='*', type=int, choices=[0,1,2,3], default=[])
parser.add_argument('--drop_interval', nargs=2, type=float)
parser.add_argument('--gate_override', type=float)
parser.add_argument('--wave_strength', type=float, default=1.)
parser.add_argument('--trace_wave', action='store_true')
parser.add_argument('--overwrite', action='store_true', help='Regenerate images in this experiment directory')
args = parser.parse_args()
if args.batch_size < 1 or args.num_inference_steps < 1:
    raise ValueError('batch_size and num_inference_steps must be positive')
if args.drop_interval and not 0 <= args.drop_interval[0] <= args.drop_interval[1] <= 1:
    raise ValueError('drop_interval requires 0 <= lo <= hi <= 1')

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_path = args.base_model_path
brushnet_path = args.brushnet_ckpt_path
wave = None if args.disable_wave else WaveConditioner.from_pretrained(
    args.wave_path or os.path.join(brushnet_path, "wave"), device).eval()

if wave is not None:
    wave.set_interventions(
        drop_bands=args.drop_bands,
        drop_scales=args.drop_scales,
        drop_interval=args.drop_interval,
        gate_override=args.gate_override,
        wave_strength=args.wave_strength,
    )

eval_dtype = TRAIN_VALIDATION_DTYPE if ALIGN_TRAIN_VALIDATION_24X2 else torch.float16
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=eval_dtype).to(device)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path,
    brushnet=brushnet,
    torch_dtype=eval_dtype,
    low_cpu_mem_usage=False,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
print("NSFW safety checker:", pipe.safety_checker)

# 使用 UniPC scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()

with open(args.mapping_file, "r") as f:
    mapping_file = json.load(f)

all_mapping_items = list(mapping_file.items())
if ALIGN_TRAIN_VALIDATION_24X2:
    if len(all_mapping_items) < TRAIN_VALIDATION_COUNT:
        raise ValueError(
            f"ALIGN_TRAIN_VALIDATION_24X2 requires at least {TRAIN_VALIDATION_COUNT} mapping items, "
            f"but mapping_file only has {len(all_mapping_items)}."
        )
    mapping_items = all_mapping_items[:TRAIN_VALIDATION_COUNT]
    evaluation_repeats = TRAIN_VALIDATION_REPEATS
else:
    mapping_items = all_mapping_items
    evaluation_repeats = 1

print(
    f"Evaluation protocol: {len(mapping_items)} unique items x {evaluation_repeats} repeat(s) "
    f"= {len(mapping_items) * evaluation_repeats} generated/evaluated images; "
    f"align_train_validation={ALIGN_TRAIN_VALIDATION_24X2}"
)

batch_size = args.batch_size
os.makedirs(args.image_save_path, exist_ok=True)


def repeat_output_path(image_path, repeat_idx):
    """Keep the original layout for single-pass mode; separate repeats in aligned 24x2 mode."""
    if evaluation_repeats == 1:
        return os.path.join(args.image_save_path, image_path)
    return os.path.join(args.image_save_path, f"repeat_{repeat_idx:02d}", image_path)


def masked_output_path(save_path):
    root, ext = os.path.splitext(save_path)
    return root + "_masked" + ext


# Do not silently mix different experiment settings in the same output directory.
protocol_path = Path(args.image_save_path) / 'evaluation_config.json'
protocol = {k: v for k, v in vars(args).items() if k != 'overwrite'}
protocol.update({
    'align_train_validation_24x2': ALIGN_TRAIN_VALIDATION_24X2,
    'evaluation_unique_items': len(mapping_items),
    'evaluation_repeats': evaluation_repeats,
    'evaluation_total_images': len(mapping_items) * evaluation_repeats,
})
if protocol_path.exists() and not args.overwrite and json.loads(protocol_path.read_text()) != protocol:
    raise ValueError('Output configuration differs; choose a new image_save_path or --overwrite')
protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))

# In aligned mode, training validation seeds ONCE and reuses one generator across all
# batches and both repeats. This is intentionally different from the original evaluator,
# which recreated per-image generators (and reset torch.manual_seed) for every batch.
if ALIGN_TRAIN_VALIDATION_24X2:
    torch.manual_seed(args.seed)
    shared_generator = torch.Generator(device=device).manual_seed(args.seed)
else:
    shared_generator = None

# Exact RNG alignment cannot be resumed from a partially generated 24x2 directory,
# because skipped batches would not consume the same RNG stream. Full cache is safe.
if ALIGN_TRAIN_VALIDATION_24X2 and not args.overwrite:
    expected_paths = []
    for repeat_idx in range(evaluation_repeats):
        for _, item in mapping_items:
            sp = repeat_output_path(item['image'], repeat_idx)
            expected_paths.extend([sp, masked_output_path(sp)])
    existing = sum(os.path.exists(path) for path in expected_paths)
    if 0 < existing < len(expected_paths):
        raise ValueError(
            'Aligned 24x2 output directory is only partially cached. '
            'Use --overwrite so the shared RNG stream exactly matches training validation.'
        )

for repeat_idx in range(evaluation_repeats):
    for batch_start in range(0, len(mapping_items), batch_size):
        batch_items = mapping_items[batch_start:batch_start + batch_size]
        batch_keys, batch_data = [], []
        captions, init_images, mask_images = [], [], []
        save_paths, masked_image_save_paths = [], []

        batch_complete = all(
            os.path.exists(repeat_output_path(item['image'], repeat_idx))
            and os.path.exists(masked_output_path(repeat_output_path(item['image'], repeat_idx)))
            for _, item in batch_items
        )

        # A partially cached batch is regenerated as a whole for stable VAE RNG ordering.
        for key, item in batch_items:
            image_path = item["image"]
            mask = item[args.mask_key]
            caption = item["caption"]

            save_path = repeat_output_path(image_path, repeat_idx)
            masked_image_save_path = masked_output_path(save_path)

            if not args.overwrite and batch_complete:
                print(f"repeat {repeat_idx}: image {key} exists! skip...")
                continue

            print(f"prepare repeat {repeat_idx}, image {key} ...")

            original_path = os.path.join(args.base_dir, image_path)
            original_bgr = cv2.imread(original_path)
            if original_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {original_path}")
            original_image = original_bgr[:, :, ::-1]
            mask_np = rle2mask(mask, (512, 512))[:, :, np.newaxis]
            masked_input = original_image * (1 - mask_np)

            init_image = Image.fromarray(masked_input.astype(np.uint8)).convert("RGB")
            mask_image = Image.fromarray((mask_np.repeat(3, -1) * 255).astype(np.uint8)).convert("RGB")

            batch_keys.append(key)
            batch_data.append(item)
            captions.append(caption)
            init_images.append(init_image)
            mask_images.append(mask_image)
            save_paths.append(save_path)
            masked_image_save_paths.append(masked_image_save_path)

        if len(batch_keys) == 0:
            continue

        print(
            f"generating repeat {repeat_idx + 1}/{evaluation_repeats}: "
            f"{batch_keys[0]} -> {batch_keys[-1]} (batch size = {len(batch_keys)})"
        )

        if ALIGN_TRAIN_VALIDATION_24X2:
            # Training validation uses one generator object for the whole 24x2 run.
            generator_arg = shared_generator
        else:
            # Original evaluator behavior.
            generator_arg = [
                torch.Generator(device=device).manual_seed(args.seed)
                for _ in range(len(batch_keys))
            ]
            torch.manual_seed(args.seed)

        trace = [] if args.trace_wave else None
        if ALIGN_TRAIN_VALIDATION_24X2 and device == "cuda":
            amp_ctx = torch.autocast("cuda", dtype=TRAIN_VALIDATION_DTYPE)
        else:
            amp_ctx = contextlib.nullcontext()

        # Training validation wraps generation in accelerator.autocast(); mirror that here.
        with torch.no_grad(), amp_ctx, wave_inference(
            pipe.brushnet, wave, init_images, mask_images, trace=trace, unet=pipe.unet
        ):
            result = pipe(
                captions,
                init_images,
                mask_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator_arg,
                **conditioning_scale_kwargs(pipe, args.paintingnet_conditioning_scale),
            )

        print(f"repeat {repeat_idx}: {batch_keys[0]} -> {batch_keys[-1]}: nsfw =", result.nsfw_content_detected)
        images = result.images
        if trace is not None:
            Path(
                args.image_save_path,
                f'wave_trace_repeat{repeat_idx:02d}_{batch_start:06d}.json'
            ).write_text(json.dumps(trace))

        for key, item, image, init_image, save_path, masked_image_save_path in zip(
            batch_keys, batch_data, images, init_images, save_paths, masked_image_save_paths
        ):
            image_path = item["image"]
            mask = item[args.mask_key]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if args.blended:
                mask_np = rle2mask(mask, (512, 512))[:, :, np.newaxis]
                image_np = np.array(image)
                original_image_np = cv2.imread(os.path.join(args.base_dir, image_path))[:, :, ::-1]

                mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
                mask_blurred = mask_blurred[:, :, np.newaxis]
                mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

                image_pasted = original_image_np * (1 - mask_np) + image_np * mask_np
                image = Image.fromarray(image_pasted.astype(image_np.dtype))

            image.save(save_path)
            init_image.save(masked_image_save_path)
            print(f"saved repeat {repeat_idx}, image {key}")

# Release generation models before loading all metric networks.
del pipe, brushnet, wave
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Evaluation
# In aligned mode this evaluates exactly the same sample count as training validation:
# first 24 items x 2 generated repeats = 48 rows.
evaluation_df = pd.DataFrame(
    columns=[
        "Image ID", "repeat", "prompt", "Image Reward", "HPS V2.1", "Aesthetic Score",
        "PSNR", "LPIPS", "MSE", "CLIP Similarity"
    ]
)
metrics_calculator = MetricsCalculator(device)

metric_columns = [
    "Image Reward", "HPS V2.1", "Aesthetic Score", "PSNR", "LPIPS", "MSE", "CLIP Similarity"
]

for repeat_idx in range(evaluation_repeats):
    for key, item in mapping_items:
        print(f"evaluating repeat {repeat_idx}, image {key} ...")
        image_path = item["image"]
        mask_rle = item[args.mask_key]
        prompt = item["caption"]
        print(prompt)

        src_image_path = os.path.join(args.base_dir, image_path)
        tgt_image_path = repeat_output_path(image_path, repeat_idx)
        src_image = Image.open(src_image_path).convert("RGB").resize((512, 512))
        tgt_image = Image.open(tgt_image_path).convert("RGB").resize((512, 512))

        evaluation_result = [key, f"repeat_{repeat_idx:02d}", prompt]
        preserve_mask = 1 - rle2mask(mask_rle, (512, 512))[:, :, np.newaxis]

        for metric in metric_columns:
            print(f"evaluating metric: {metric}")

            if metric == "Image Reward":
                metric_result = metrics_calculator.calculate_image_reward(tgt_image, prompt)
            elif metric == "HPS V2.1":
                metric_result = metrics_calculator.calculate_hpsv21_score(tgt_image, prompt)
            elif metric == "Aesthetic Score":
                metric_result = metrics_calculator.calculate_aesthetic_score(tgt_image)
            elif metric == "PSNR":
                metric_result = metrics_calculator.calculate_psnr(src_image, tgt_image, preserve_mask)
            elif metric == "LPIPS":
                metric_result = metrics_calculator.calculate_lpips(src_image, tgt_image, preserve_mask)
            elif metric == "MSE":
                metric_result = metrics_calculator.calculate_mse(src_image, tgt_image, preserve_mask)
            elif metric == "CLIP Similarity":
                metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)
            else:
                raise RuntimeError(f"Unknown metric: {metric}")

            evaluation_result.append(metric_result)
            print(metric_result)

        evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

print("The averaged evaluation result:")
averaged_results = evaluation_df[metric_columns].mean(numeric_only=True)
print(averaged_results)
averaged_results.to_csv(os.path.join(args.image_save_path, "evaluation_result_sum.csv"))
evaluation_df.to_csv(os.path.join(args.image_save_path, "evaluation_result.csv"), index=False)
print(
    f"The generated images and evaluation results are saved in {args.image_save_path}; "
    f"rows={len(evaluation_df)}"
)
