from wavebrush.core import WaveConditioner
from wavebrush.integration import wave_inference, conditioning_scale_kwargs
import gc
import argparse
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


# BrushNet-official evaluation protocol, with ONE intentional model deviation:
# keep the user's RealisticVision base model instead of runwayml/stable-diffusion-v1-5.
# Official generation settings reproduced here:
#   * full mapping_file, one pass
#   * batch size = 1
#   * fresh torch.Generator(device).manual_seed(1234) for every image
#   * fp16 BrushNet + pipeline
#   * UniPC scheduler
#   * 50 denoising steps
#   * guidance_scale = 7.5 (pipeline default, passed explicitly)
OFFICIAL_BATCH_SIZE = 1
OFFICIAL_SEED = 1234
OFFICIAL_NUM_INFERENCE_STEPS = 50
OFFICIAL_GUIDANCE_SCALE = 7.5
OFFICIAL_DTYPE = torch.float16


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

        # Same metric definitions/backbones as BrushNet official evaluator.
        self.clip_metric_calculator = CLIPScore(
            model_name_or_path="openai/clip-vit-large-patch14"
        ).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze"
        ).to(device)

        # Same LAION aesthetic predictor, loaded from the user's local checkpoints.
        self.aesthetic_model = torch.nn.Linear(768, 1)
        aesthetic_model_ckpt_path = os.path.join(
            ckpt_path, "aesthetic", "sa_0_4_vit_l_14_linear.pth"
        )
        self.aesthetic_model.load_state_dict(
            torch.load(aesthetic_model_ckpt_path, map_location="cpu")
        )
        self.aesthetic_model = self.aesthetic_model.to(device).eval()

        openclip_path = os.path.join(ckpt_path, "aesthetic", "ViT-L-14.pt")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=openclip_path, load_weights_only=False
        )
        self.clip_model = self.clip_model.to(device).eval()

        # Same ImageReward model family as official evaluator, loaded locally.
        image_reward_path = os.path.join(ckpt_path, "ImageReward", "ImageReward.pt")
        image_reward_med_config = os.path.join(ckpt_path, "ImageReward", "med_config.json")
        self.imagereward_model = RM.load(
            image_reward_path, device=device, med_config=image_reward_med_config
        )

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
        # IMPORTANT: this intentionally matches BrushNet's published evaluator.
        # RGB channels are summed in the numerator while the denominator is mask.sum(),
        # i.e. there is NO extra division by 3.
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
        # Same non-channel-averaged MSE definition as BrushNet official evaluator.
        difference_size = mask.sum()
        mse = difference_square_sum / difference_size
        return mse.item()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--brushnet_ckpt_path",
    type=str,
    default="data/ckpt/segmentation_mask_brushnet_ckpt",
)
# Intentionally NOT changed to runwayml/stable-diffusion-v1-5.
parser.add_argument(
    "--base_model_path",
    type=str,
    default="data/ckpt/realisticVisionV60B1_v51VAE",
)
parser.add_argument(
    "--image_save_path",
    type=str,
    default="runs/evaluation_result/BrushBench/brushnet_segmask/inside",
)
parser.add_argument("--mapping_file", type=str, default="data/BrushBench/mapping_file.json")
parser.add_argument("--base_dir", type=str, default="data/BrushBench")
parser.add_argument("--mask_key", type=str, default="inpainting_mask")
parser.add_argument("--blended", action="store_true")
parser.add_argument("--paintingnet_conditioning_scale", type=float, default=1.0)
# Kept as an argument for compatibility with the scan runner, but official protocol requires 1.
parser.add_argument("--batch_size", type=int, default=OFFICIAL_BATCH_SIZE)
parser.add_argument("--wave_path", type=str, default=None, help="Wave folder; default: brushnet_ckpt_path/wave")
parser.add_argument("--disable_wave", action="store_true", help="Evaluate original BrushNet baseline")
parser.add_argument("--seed", type=int, default=OFFICIAL_SEED)
parser.add_argument("--num_inference_steps", type=int, default=OFFICIAL_NUM_INFERENCE_STEPS)
parser.add_argument("--guidance_scale", type=float, default=OFFICIAL_GUIDANCE_SCALE)
parser.add_argument("--metric_ckpt_path", default="data/ckpt")
parser.add_argument("--drop_bands", nargs="*", choices=["H1", "H2", "H3", "L3"], default=[])
parser.add_argument("--drop_scales", nargs="*", type=int, choices=[0, 1, 2, 3], default=[])
parser.add_argument("--drop_interval", nargs=2, type=float)
parser.add_argument("--gate_override", type=float)
parser.add_argument("--wave_strength", type=float, default=1.0)
parser.add_argument("--wave_band_gains", type=float, nargs=4, default=None, metavar=("H1", "H2", "H3", "L3"))
parser.add_argument("--wave_stage_gains", type=float, nargs=5, default=None, metavar=("S0", "S1", "S2", "S3", "MID"))
parser.add_argument(
    "--wave_time_gains",
    type=float,
    nargs=10,
    default=None,
    metavar=("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"),
)
parser.add_argument(
    "--wave_band_time_gains",
    type=float,
    nargs=40,
    default=None,
    help="Flattened 4x10 Band x Time matrix: H1 row, H2 row, H3 row, L3 row",
)
parser.add_argument(
    "--wave_band_stage_gains",
    type=float,
    nargs=20,
    default=None,
    help="Flattened 4x5 Band x Stage matrix: H1 row, H2 row, H3 row, L3 row",
)
parser.add_argument("--trace_wave", action="store_true")
parser.add_argument("--overwrite", action="store_true", help="Regenerate images in this experiment directory")
args = parser.parse_args()

if args.batch_size != OFFICIAL_BATCH_SIZE:
    raise ValueError(
        f"BrushNet official evaluation uses batch_size={OFFICIAL_BATCH_SIZE}; "
        f"got --batch_size {args.batch_size}. Remove the argument or set it to 1."
    )
if args.num_inference_steps < 1:
    raise ValueError("num_inference_steps must be positive")
if args.drop_interval and not 0 <= args.drop_interval[0] <= args.drop_interval[1] <= 1:
    raise ValueError("drop_interval requires 0 <= lo <= hi <= 1")

# Make deviations from the official protocol explicit instead of silently changing the run.
if args.seed != OFFICIAL_SEED:
    print(f"[WARN] official BrushNet seed is {OFFICIAL_SEED}; current seed={args.seed}")
if args.num_inference_steps != OFFICIAL_NUM_INFERENCE_STEPS:
    print(
        f"[WARN] official BrushNet num_inference_steps is {OFFICIAL_NUM_INFERENCE_STEPS}; "
        f"current={args.num_inference_steps}"
    )
if abs(args.guidance_scale - OFFICIAL_GUIDANCE_SCALE) > 1e-12:
    print(
        f"[WARN] official/pipeline-default guidance_scale is {OFFICIAL_GUIDANCE_SCALE}; "
        f"current={args.guidance_scale}"
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_path = args.base_model_path
brushnet_path = args.brushnet_ckpt_path

wave = None if args.disable_wave else WaveConditioner.from_pretrained(
    args.wave_path or os.path.join(brushnet_path, "wave"), device
).eval()

if wave is not None:
    wave.set_interventions(
        drop_bands=args.drop_bands,
        drop_scales=args.drop_scales,
        drop_interval=args.drop_interval,
        gate_override=args.gate_override,
        wave_strength=args.wave_strength,
        band_gains=args.wave_band_gains,
        stage_gains=args.wave_stage_gains,
        time_gains=args.wave_time_gains,
        band_time_gains=args.wave_band_time_gains,
        band_stage_gains=args.wave_band_stage_gains,
    )

# Official BrushNet evaluator uses fp16 for both BrushNet and the SD pipeline.
eval_dtype = OFFICIAL_DTYPE
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=eval_dtype).to(device)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path,
    brushnet=brushnet,
    torch_dtype=eval_dtype,
    low_cpu_mem_usage=False,
    # BrushNet README asks users to ignore NSFW filtering for correct benchmark results.
    # Disabling the checker here is the local equivalent and avoids changing generated images.
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

with open(args.mapping_file, "r") as f:
    mapping_file = json.load(f)

# Official evaluator uses the complete mapping_file exactly once.
mapping_items = list(mapping_file.items())
evaluation_repeats = 1
print(
    f"Evaluation protocol: full mapping_file, {len(mapping_items)} images x 1 pass; "
    f"batch_size=1, seed reset per image, dtype=fp16, steps={args.num_inference_steps}, "
    f"base_model={base_model_path}"
)

os.makedirs(args.image_save_path, exist_ok=True)

# Persist the exact protocol next to outputs so routing sweeps cannot silently mix settings.
protocol_path = Path(args.image_save_path) / "evaluation_config.json"
protocol = {k: v for k, v in vars(args).items() if k != "overwrite"}
protocol.update(
    {
        "protocol": "brushnet_official_generation_except_base_model",
        "official_batch_size": OFFICIAL_BATCH_SIZE,
        "seed_reset_per_image": True,
        "evaluation_unique_items": len(mapping_items),
        "evaluation_repeats": 1,
        "evaluation_total_images": len(mapping_items),
        "eval_dtype": "float16",
    }
)
if protocol_path.exists() and not args.overwrite:
    old_protocol = json.loads(protocol_path.read_text())
    if old_protocol != protocol:
        raise ValueError("Output configuration differs; choose a new image_save_path or --overwrite")
protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))


def masked_output_path(save_path):
    root, ext = os.path.splitext(save_path)
    return root + "_masked" + ext


# Generation: deliberately single-image, like the official BrushNet evaluator.
for image_index, (key, item) in enumerate(mapping_items):
    print(f"generating image {key} ...")
    image_path = item["image"]
    mask = item[args.mask_key]
    caption = item["caption"]

    save_path = os.path.join(args.image_save_path, image_path)
    masked_image_save_path = masked_output_path(save_path)

    if not args.overwrite and os.path.exists(save_path) and os.path.exists(masked_image_save_path):
        print(f"image {key} exists! skip...")
        continue

    original_path = os.path.join(args.base_dir, image_path)
    original_bgr = cv2.imread(original_path)
    if original_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {original_path}")
    original_image = original_bgr[:, :, ::-1]
    mask_np = rle2mask(mask, (512, 512))[:, :, np.newaxis]
    masked_input = original_image * (1 - mask_np)

    init_image = Image.fromarray(masked_input.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray((mask_np.repeat(3, -1) * 255).astype(np.uint8)).convert("RGB")

    # Critical official behavior: recreate the generator and reset it to seed 1234
    # for EVERY image, instead of sharing one RNG stream across the dataset.
    generator = torch.Generator(device=device).manual_seed(args.seed)

    trace = [] if args.trace_wave else None
    with torch.no_grad(), wave_inference(
        pipe.brushnet,
        wave,
        [init_image],
        [mask_image],
        trace=trace,
        unet=pipe.unet,
    ):
        result = pipe(
            caption,
            init_image,
            mask_image,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            **conditioning_scale_kwargs(pipe, args.paintingnet_conditioning_scale),
        )

    print(f"image {key}: nsfw =", result.nsfw_content_detected)
    image = result.images[0]

    if trace is not None:
        Path(
            args.image_save_path,
            f"wave_trace_{image_index:06d}.json",
        ).write_text(json.dumps(trace))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if args.blended:
        blend_mask = rle2mask(mask, (512, 512))[:, :, np.newaxis]
        image_np = np.array(image)
        original_image_np = cv2.imread(original_path)[:, :, ::-1]

        mask_blurred = cv2.GaussianBlur(blend_mask * 255, (21, 21), 0) / 255
        mask_blurred = mask_blurred[:, :, np.newaxis]
        blend_mask = 1 - (1 - blend_mask) * (1 - mask_blurred)

        image_pasted = original_image_np * (1 - blend_mask) + image_np * blend_mask
        image = Image.fromarray(image_pasted.astype(image_np.dtype))

    image.save(save_path)
    init_image.save(masked_image_save_path)
    print(f"saved image {key}")

# Release generation models before loading the metric networks.
del pipe, brushnet, wave
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Evaluation: full mapping_file, once, matching official BrushNet evaluator.
evaluation_df = pd.DataFrame(
    columns=[
        "Image ID",
        "Image Reward",
        "HPS V2.1",
        "Aesthetic Score",
        "PSNR",
        "LPIPS",
        "MSE",
        "CLIP Similarity",
    ]
)
metrics_calculator = MetricsCalculator(device, args.metric_ckpt_path)
metric_columns = [
    "Image Reward",
    "HPS V2.1",
    "Aesthetic Score",
    "PSNR",
    "LPIPS",
    "MSE",
    "CLIP Similarity",
]

for key, item in mapping_items:
    print(f"evaluating image {key} ...")
    image_path = item["image"]
    mask_rle = item[args.mask_key]
    prompt = item["caption"]
    print(prompt)

    src_image_path = os.path.join(args.base_dir, image_path)
    tgt_image_path = os.path.join(args.image_save_path, image_path)
    src_image = Image.open(src_image_path).convert("RGB").resize((512, 512))
    tgt_image = Image.open(tgt_image_path).convert("RGB").resize((512, 512))

    evaluation_result = [key]
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
