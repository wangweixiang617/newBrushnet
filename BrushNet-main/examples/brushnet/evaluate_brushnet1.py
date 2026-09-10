import argparse
import json
import math
import os

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
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_path = args.base_model_path
brushnet_path = args.brushnet_ckpt_path

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16).to(device)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path,
    brushnet=brushnet,
    torch_dtype=torch.float16,
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

mapping_items = list(mapping_file.items())
batch_size = args.batch_size

for batch_start in range(0, len(mapping_items), batch_size):
    batch_items = mapping_items[batch_start:batch_start + batch_size]
    batch_keys, batch_data = [], []
    captions, init_images, mask_images = [], [], []
    save_paths, masked_image_save_paths = [], []

    # 准备一个 batch
    for key, item in batch_items:
        image_path = item["image"]
        mask = item[args.mask_key]
        caption = item["caption"]

        save_path = os.path.join(args.image_save_path, image_path)
        root, ext = os.path.splitext(save_path)
        masked_image_save_path = root + "_masked" + ext

        # 已经生成过则跳过
        if os.path.exists(save_path) and os.path.exists(masked_image_save_path):
            print(f"image {key} exists! skip...")
            continue

        print(f"prepare image {key} ...")

        # 原图 / mask / masked input
        original_image = cv2.imread(os.path.join(args.base_dir, image_path))[:, :, ::-1]
        mask_np = rle2mask(mask, (512, 512))[:, :, np.newaxis]
        masked_input = original_image * (1 - mask_np)

        gt_image = Image.fromarray(original_image.astype(np.uint8)).convert("RGB")
        init_image = Image.fromarray(masked_input.astype(np.uint8)).convert("RGB")
        mask_image = Image.fromarray((mask_np.repeat(3, -1) * 255).astype(np.uint8)).convert("RGB")

        #保存原图和mask图
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # root, ext = os.path.splitext(save_path)
        # ori_image_save_path = root + "_ori" + ext
        # mask_image_save_path = root + "_mask" + ext
        # gt_image.save(ori_image_save_path)
        # mask_image.save(mask_image_save_path)

        batch_keys.append(key)
        batch_data.append(item)
        captions.append(caption)
        init_images.append(init_image)
        mask_images.append(mask_image)
        save_paths.append(save_path)
        masked_image_save_paths.append(masked_image_save_path)

    if len(batch_keys) == 0:
        continue

    print(f"generating batch: {batch_keys[0]} -> {batch_keys[-1]} (batch size = {len(batch_keys)})")

    # 每张图一个固定 seed generator
    generators = [torch.Generator(device=device).manual_seed(1234) for _ in range(len(batch_keys))]

    # Batch inference
    result = pipe(
        captions,
        init_images,
        mask_images,
        num_inference_steps=50,
        generator=generators,
        paintingnet_conditioning_scale=args.paintingnet_conditioning_scale,
    )
    print(f"{batch_keys[0]} -> {batch_keys[-1]}: nsfw =", result.nsfw_content_detected)
    images = result.images

    # 保存 batch 中每一张图片
    for key, item, image, init_image, save_path, masked_image_save_path in zip(
        batch_keys, batch_data, images, init_images, save_paths, masked_image_save_paths
    ):
        image_path = item["image"]
        mask = item[args.mask_key]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # blended 模式
        if args.blended:
            mask_np = rle2mask(mask, (512, 512))[:, :, np.newaxis]
            image_np = np.array(image)
            original_image_np = cv2.imread(os.path.join(args.base_dir, image_path))[:, :, ::-1]

            # blur
            mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
            mask_blurred = mask_blurred[:, :, np.newaxis]
            mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

            image_pasted = original_image_np * (1 - mask_np) + image_np * mask_np
            image = Image.fromarray(image_pasted.astype(image_np.dtype))

        image.save(save_path)
        init_image.save(masked_image_save_path)
        print(f"saved image {key}")

# Evaluation
evaluation_df = pd.DataFrame(
    columns=["Image ID","prompt", "Image Reward", "HPS V2.1", "Aesthetic Score", "PSNR", "LPIPS", "MSE", "CLIP Similarity"]
)
metrics_calculator = MetricsCalculator(device)

for key, item in mapping_file.items():
    print(f"evaluating image {key} ...")
    image_path = item["image"]
    mask = item[args.mask_key]
    prompt = item["caption"]
    print(prompt)

    src_image_path = os.path.join(args.base_dir, image_path)
    tgt_image_path = os.path.join(args.image_save_path, image_path)
    src_image = Image.open(src_image_path).resize((512, 512))
    tgt_image = Image.open(tgt_image_path).resize((512, 512))

    evaluation_result = [key]
    evaluation_result.append(prompt)
    mask = 1 - rle2mask(mask, (512, 512))[:, :, np.newaxis]

    for metric in evaluation_df.columns.values.tolist()[2:]:
        print(f"evluating metric: {metric}")

        if metric == "Image Reward":
            metric_result = metrics_calculator.calculate_image_reward(tgt_image, prompt)
        elif metric == "HPS V2.1":
            metric_result = metrics_calculator.calculate_hpsv21_score(tgt_image, prompt)
        elif metric == "Aesthetic Score":
            metric_result = metrics_calculator.calculate_aesthetic_score(tgt_image)
        elif metric == "PSNR":
            metric_result = metrics_calculator.calculate_psnr(src_image, tgt_image, mask)
        elif metric == "LPIPS":
            metric_result = metrics_calculator.calculate_lpips(src_image, tgt_image, mask)
        elif metric == "MSE":
            metric_result = metrics_calculator.calculate_mse(src_image, tgt_image, mask)
        elif metric == "CLIP Similarity":
            metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)

        evaluation_result.append(metric_result)
        print(metric_result)

    evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

print("The averaged evaluation result:")
averaged_results = evaluation_df.mean(numeric_only=True)
print(averaged_results)
averaged_results.to_csv(os.path.join(args.image_save_path, "evaluation_result_sum.csv"))
evaluation_df.to_csv(os.path.join(args.image_save_path, "evaluation_result.csv"))
print(f"The generated images and evaluation results is saved in {args.image_save_path}")
