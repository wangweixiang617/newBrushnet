import contextlib
import gc
import math
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import open_clip
import ImageReward as RM


class BrushNetValidationEvaluator:
    """
    BrushNet validation evaluator.

    light mode:
        PSNR
        LPIPS
        MSE

    full mode:
        Image Reward
        HPS V2.1
        Aesthetic Score
        PSNR
        LPIPS
        MSE
        CLIP Similarity

    BrushNet preservation metrics:
        原始 mask:
            1 = 编辑区域
            0 = 保留区域

        评价:
            preserve_mask = 1 - mask

        因此 PSNR / LPIPS / MSE 都只评价非编辑区域。

    显存策略（offload=True）:
        - metric 默认保存在 CPU
        - 使用时 CPU -> GPU
        - 当前 metric 计算结束后 GPU -> CPU
        - 避免 LPIPS / CLIP / Aesthetic / ImageReward / HPS 同时常驻 GPU

    HPS 优化:
        不再直接调用 hpsv2.score()。官方 hpsv2.score() 每次调用都会重新
        torch.load(checkpoint) + load_state_dict()。

        当前实现:
            第一次 full validation:
                创建 ViT-H-14
                checkpoint 只加载一次
                模型保存在 CPU

            每次 HPS:
                CPU -> GPU
                计算所有 validation samples
                GPU -> CPU

        后续 full validation 不再重新读取 HPS checkpoint。
    """

    def __init__(
        self,
        device,
        ckpt_path="data/ckpt",
        clip_score_model_path="openai/clip-vit-large-patch14",
        hps_ckpt_path=None,
        hps_batch_size=1,
        offload=True,
    ):
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path
        self.clip_score_model_path = clip_score_model_path
        self.hps_ckpt_path = hps_ckpt_path

        # HPS ViT-H-14 很大；训练时默认 batch=1 更稳，显存足够可改 2 / 4。
        self.hps_batch_size = max(1, int(hps_batch_size))

        # True: 每种 metric 用完以后移回 CPU。训练阶段建议保持 True。
        self.offload = offload

        # Light metric ---------------------------------------------------------
        # LPIPS 使用 BrushNet 官方一致的 SqueezeNet；初始化时默认在 CPU。
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").eval()

        # Heavy metrics --------------------------------------------------------
        # 全部 lazy load，第一次 full=True 时才加载。
        self.clip_metric = None
        self.aesthetic_model = None
        self.aesthetic_clip_model = None
        self.aesthetic_preprocess = None
        self.image_reward_model = None

        # HPS v2.1: lazy load on CPU；checkpoint 只加载一次，full validation 时 CPU <-> GPU。
        self.hps_model = None
        self.hps_preprocess = None
        self.hps_tokenizer = None
        self._resolved_hps_ckpt_path = None

    # Basic utilities =========================================================

    @staticmethod
    def _clear_cuda():
        """
        清理 Python 无引用对象和 PyTorch CUDA cache。

        注意: empty_cache() 不能释放仍被 Tensor/model 引用的显存，
        所以必须先 .cpu() 或 del。
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _to_float(value):
        """将各种 metric 输出统一转换成 Python float。"""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return float("nan")
            return BrushNetValidationEvaluator._to_float(value[0])
        return float(value)

    # Image / mask ============================================================

    @staticmethod
    def _prepare_images(gt_image, pred_image):
        """PIL -> numpy float32 [0,1]."""
        gt_image = gt_image.convert("RGB")
        pred_image = pred_image.convert("RGB")

        if pred_image.size != gt_image.size:
            pred_image = pred_image.resize(gt_image.size, Image.Resampling.BILINEAR)

        gt = np.asarray(gt_image).astype(np.float32) / 255.0
        pred = np.asarray(pred_image).astype(np.float32) / 255.0
        return gt, pred

    @staticmethod
    def _prepare_preserve_mask(mask_image, image_size):
        """
        BrushNet:
            白色 / 1 = 编辑区域
            黑色 / 0 = 非编辑区域

        preservation metric:
            preserve_mask = 1 - edit_mask
        """
        mask_image = mask_image.convert("L")

        if mask_image.size != image_size:
            mask_image = mask_image.resize(image_size, Image.Resampling.NEAREST)

        mask = np.asarray(mask_image).astype(np.float32) / 255.0

        # 二值化，避免 resize 等操作产生灰度 mask。
        mask = (mask > 0.5).astype(np.float32)
        preserve_mask = 1.0 - mask
        return preserve_mask[:, :, None]

    # PSNR / MSE ==============================================================

    @staticmethod
    def _calculate_mse(gt, pred, preserve_mask):
        """
        与 BrushNet 官方 evaluation 的 MSE 定义保持一致。

        注意:
            numerator: RGB 3 channel 全部求和
            denominator: preserve_mask.sum()

        denominator 没有乘 3，不要改成常规 RGB channel mean，
        否则数值会和 BrushNet 官方 evaluation 不一致。
        """
        gt_masked = gt * preserve_mask
        pred_masked = pred * preserve_mask
        difference = pred_masked - gt_masked

        difference_square_sum = (difference ** 2).sum()
        difference_size = preserve_mask.sum()

        if difference_size <= 0:
            return 0.0
        return float(difference_square_sum / difference_size)

    @staticmethod
    def _calculate_psnr(mse):
        """BrushNet 官方 PSNR 逻辑。"""
        if mse < 1.0e-10:
            return 1000.0
        return 20.0 * math.log10(1.0 / math.sqrt(mse))

    # LPIPS ===================================================================

    def _lpips_to_gpu(self):
        self.lpips_metric = self.lpips_metric.to(self.device).eval()

    def _lpips_to_cpu(self):
        if self.offload and self.lpips_metric is not None:
            self.lpips_metric = self.lpips_metric.cpu()
        self._clear_cuda()

    def _calculate_lpips_one(self, gt, pred, preserve_mask):
        """
        此函数假设 LPIPS 已经在 GPU。

        与 BrushNet 官方 evaluation 一致:
            image -> [0,1]
            mask
            -> [-1,1]
            -> LPIPS SqueezeNet
        """
        gt = gt * preserve_mask
        pred = pred * preserve_mask

        gt_tensor = pred_tensor = score = None
        try:
            gt_tensor = (
                torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
            )
            pred_tensor = (
                torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
            )

            # [0,1] -> [-1,1]
            gt_tensor = gt_tensor * 2.0 - 1.0
            pred_tensor = pred_tensor * 2.0 - 1.0

            self.lpips_metric.reset()
            with torch.inference_mode():
                score = self.lpips_metric(pred_tensor, gt_tensor)

            return float(score.detach().cpu().item())
        finally:
            # 单张中间 Tensor 显式释放；整个 batch 完成后 LPIPS model 再回 CPU。
            del gt_tensor, pred_tensor, score

    # CLIP Similarity =========================================================

    def _load_clip_metric(self):
        """CLIPScore lazy load；第一次 full=True 才创建，默认留在 CPU。"""
        if self.clip_metric is None:
            print("[Evaluator] Loading CLIPScore...")
            self.clip_metric = CLIPScore(model_name_or_path=self.clip_score_model_path).eval()

    def _calculate_clip_batch(self, samples):
        """
        CLIP 生命周期:
            CPU -> GPU -> 所有 samples -> CPU

        CPU -> GPU 也放入 try；即使 .to(self.device) 异常，finally 仍会尝试清理。
        """
        self._load_clip_metric()
        scores = []

        try:
            self.clip_metric = self.clip_metric.to(self.device).eval()

            for sample in samples:
                image_tensor = score = None
                try:
                    image_np = np.asarray(sample["pred_image"].convert("RGB"))

                    # 与 BrushNet 官方 CLIPScore 输入形式一致: uint8 HWC -> CHW Tensor。
                    image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).to(self.device)

                    self.clip_metric.reset()
                    with torch.inference_mode():
                        score = self.clip_metric(image_tensor, sample["prompt"])

                    scores.append(float(score.detach().cpu().item()))
                finally:
                    del image_tensor, score
        finally:
            if self.offload and self.clip_metric is not None:
                self.clip_metric = self.clip_metric.cpu()
            self._clear_cuda()

        return scores

    # Aesthetic Score =========================================================

    def _load_aesthetic_model(self):
        """
        Aesthetic:
            OpenCLIP ViT-L/14 + 768 -> 1 linear predictor

        第一次 full=True 才加载。
        """
        if self.aesthetic_model is not None:
            return

        print("[Evaluator] Loading Aesthetic model...")

        linear_path = os.path.join(self.ckpt_path, "aesthetic", "sa_0_4_vit_l_14_linear.pth")
        openclip_path = os.path.join(self.ckpt_path, "aesthetic", "ViT-L-14.pt")

        if not os.path.isfile(linear_path) or not os.path.isfile(openclip_path):
            raise FileNotFoundError(
                f"Aesthetic checkpoint missing: {linear_path} / {openclip_path}"
            )

        # linear head 初始化在 CPU。
        self.aesthetic_model = torch.nn.Linear(768, 1)
        state_dict = torch.load(linear_path, map_location="cpu")
        self.aesthetic_model.load_state_dict(state_dict)
        del state_dict
        self.aesthetic_model.eval()

        # OpenAI CLIP ViT-L/14；当前 torch 2.11 环境使用 load_weights_only=False。
        self.aesthetic_clip_model, _, self.aesthetic_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=openclip_path,
            load_weights_only=False,
        )
        self.aesthetic_clip_model.eval()

    def _calculate_aesthetic_batch(self, samples):
        """
        Aesthetic 生命周期:
            Linear CPU -> GPU
            ViT-L CPU -> GPU
            所有 samples
            两个模型 -> CPU

        两个 .to(cuda) 都放入 try，避免一个已上 GPU、另一个 OOM 后无法清理。
        """
        self._load_aesthetic_model()
        scores = []

        try:
            self.aesthetic_model = self.aesthetic_model.to(self.device).eval()
            self.aesthetic_clip_model = self.aesthetic_clip_model.to(self.device).eval()

            with torch.inference_mode():
                for sample in samples:
                    image = image_features = prediction = None
                    try:
                        image = (
                            self.aesthetic_preprocess(sample["pred_image"].convert("RGB"))
                            .unsqueeze(0)
                            .to(self.device)
                        )

                        image_features = self.aesthetic_clip_model.encode_image(image)

                        # BrushNet 官方 aesthetic predictor: image feature 先做 L2 normalization。
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        prediction = self.aesthetic_model(image_features)

                        scores.append(float(prediction.detach().cpu().item()))
                    finally:
                        del image, image_features, prediction
        finally:
            if self.offload:
                if self.aesthetic_model is not None:
                    self.aesthetic_model = self.aesthetic_model.cpu()
                if self.aesthetic_clip_model is not None:
                    self.aesthetic_clip_model = self.aesthetic_clip_model.cpu()

            self._clear_cuda()

        return scores

    # ImageReward ==============================================================

    def _load_image_reward(self):
        """ImageReward lazy load；第一次 full=True 时加载到 CPU。"""
        if self.image_reward_model is not None:
            return

        print("[Evaluator] Loading ImageReward...")

        image_reward_path = os.path.join(self.ckpt_path, "ImageReward", "ImageReward.pt")
        med_config_path = os.path.join(self.ckpt_path, "ImageReward", "med_config.json")

        if not os.path.isfile(image_reward_path) or not os.path.isfile(med_config_path):
            raise FileNotFoundError(
                f"ImageReward checkpoint missing: {image_reward_path} / {med_config_path}"
            )

        # 初始化到 CPU。
        self.image_reward_model = RM.load(
            image_reward_path,
            device="cpu",
            med_config=med_config_path,
        ).eval()

    def _image_reward_to_device(self, device):
        """
        ImageReward 不仅要 model.to(device)。

        它内部 score() 还使用 self.device 移动 tokenizer/image tensor，
        因此必须同步修改 self.device。
        """
        self.image_reward_model = self.image_reward_model.to(device)
        self.image_reward_model.device = torch.device(device)

    def _calculate_image_reward_batch(self, samples):
        """
        ImageReward:
            CPU -> GPU -> 所有 samples -> CPU
        """
        self._load_image_reward()
        scores = []

        try:
            self._image_reward_to_device(self.device)
            self.image_reward_model.eval()

            with torch.inference_mode():
                for sample in samples:
                    reward = self.image_reward_model.score(
                        sample["prompt"],
                        [sample["pred_image"]],
                    )
                    scores.append(self._to_float(reward))
        finally:
            if self.offload and self.image_reward_model is not None:
                try:
                    self._image_reward_to_device("cpu")
                except Exception as e:
                    print("[Evaluator] ImageReward offload warning:", e)

            self._clear_cuda()

        return scores

    # HPS v2.1 ================================================================

    def _resolve_hps_checkpoint(self):
        """
        HPS v2.1 checkpoint 路径优先级:

            1. 显式传入 hps_ckpt_path
            2. Hugging Face cache / 当前 HF_ENDPOINT

        路径只解析一次。
        """
        if self._resolved_hps_ckpt_path is not None:
            return self._resolved_hps_ckpt_path

        if self.hps_ckpt_path is not None:
            path = os.path.expanduser(self.hps_ckpt_path)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"HPS checkpoint not found: {path}")
        else:
            from huggingface_hub import hf_hub_download
            from hpsv2.utils import hps_version_map

            print("[Evaluator] Resolving HPS v2.1 checkpoint from Hugging Face cache / configured endpoint...")
            path = hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])

        self._resolved_hps_ckpt_path = path
        return path

    def _load_hps_model(self):
        """
        HPS 模型和 checkpoint 只初始化一次。

        第一次 full=True:
            create ViT-H-14 on CPU
            -> torch.load(checkpoint, map_location="cpu")
            -> load_state_dict()
            -> tokenizer
            -> 保留 CPU 模型

        后续 full=True:
            直接复用 self.hps_model，不重新读取 checkpoint。
        """
        if self.hps_model is not None:
            return

        print("[Evaluator] Loading HPS v2.1 model/checkpoint once on CPU...")

        from hpsv2.src.open_clip import (
            create_model_and_transforms as hps_create_model_and_transforms,
            get_tokenizer as hps_get_tokenizer,
        )

        self.hps_model, _, self.hps_preprocess = hps_create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",  # 与 HPSv2 官方参数保持一致。
            device="cpu",
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        # 关键优化: HPS checkpoint 只在这里 torch.load 一次。
        checkpoint = torch.load(self._resolve_hps_checkpoint(), map_location="cpu")
        try:
            self.hps_model.load_state_dict(checkpoint["state_dict"])
        finally:
            del checkpoint

        self.hps_tokenizer = hps_get_tokenizer("ViT-H-14")
        self.hps_model.eval()
        gc.collect()

    def _calculate_hps_batch(self, samples):
        """
        优化后的 HPS v2.1:

        - 不调用 hpsv2.score()
        - checkpoint 只加载一次
        - 当前 full validation: CPU -> GPU -> 所有 HPS 样本 -> CPU
        - 后续 full validation 复用 CPU 中的模型
        - 相同 prompt 分组，再按 hps_batch_size 分块控制 ViT-H-14 显存峰值
        """
        self._load_hps_model()

        grouped = defaultdict(list)
        for index, sample in enumerate(samples):
            grouped[sample["prompt"]].append((index, sample["pred_image"]))

        results = [None] * len(samples)

        try:
            # CPU -> GPU 也放入 try。
            self.hps_model = self.hps_model.to(self.device).eval()

            with torch.inference_mode():
                for prompt, items in grouped.items():
                    for start in range(0, len(items), self.hps_batch_size):
                        chunk = items[start : start + self.hps_batch_size]

                        image_batch = text_batch = outputs = logits_per_image = chunk_scores = None
                        image_features = text_features = chunk_scores_cpu = None

                        try:
                            image_tensors = [
                                self.hps_preprocess(image.convert("RGB"))
                                for _, image in chunk
                            ]

                            image_batch = torch.stack(image_tensors, dim=0).to(
                                self.device,
                                non_blocking=True,
                            )

                            # 每张图对应同一个 prompt，与官方逐图 hpsv2.score() 语义一致。
                            text_batch = self.hps_tokenizer([prompt] * len(chunk)).to(
                                self.device,
                                non_blocking=True,
                            )

                            # HPS 官方 CUDA 推理使用 autocast。
                            amp_ctx = (
                                torch.autocast("cuda")
                                if self.device.type == "cuda"
                                else contextlib.nullcontext()
                            )

                            with amp_ctx:
                                outputs = self.hps_model(image_batch, text_batch)
                                image_features = outputs["image_features"]
                                text_features = outputs["text_features"]
                                logits_per_image = image_features @ text_features.T

                                # 每个 image 与同 index text 配对，所以取 diagonal。
                                chunk_scores = torch.diagonal(logits_per_image)

                            chunk_scores_cpu = (
                                chunk_scores.float().detach().cpu().tolist()
                            )

                            for (item_index, _), score in zip(chunk, chunk_scores_cpu):
                                results[item_index] = float(score)

                            del image_tensors
                        finally:
                            del image_batch, text_batch, outputs, logits_per_image, chunk_scores
                            del image_features, text_features, chunk_scores_cpu
        finally:
            if self.offload and self.hps_model is not None:
                try:
                    self.hps_model = self.hps_model.cpu()
                except Exception as e:
                    print("[Evaluator] HPS offload warning:", e)

            self._clear_cuda()

        if any(score is None for score in results):
            raise RuntimeError("HPS evaluation did not produce a score for every sample.")

        return results

    # Public API ===============================================================

    def evaluate(self, gt_image, pred_image, mask_image, prompt=None, full=False):
        """
        单张接口。

        full=False:
            PSNR
            LPIPS
            MSE

        full=True:
            7 个指标

        单张也走 evaluate_batch()，因此 offload=True 时不会留下 GPU metric model。
        """
        sample = {
            "gt_image": gt_image,
            "pred_image": pred_image,
            "mask_image": mask_image,
            "prompt": prompt,
        }
        return self.evaluate_batch([sample], full=full)

    def evaluate_batch(self, samples, full=False):
        """
        推荐训练 validation 使用。

        samples:
            [
                {
                    "gt_image": PIL.Image,
                    "pred_image": PIL.Image,
                    "mask_image": PIL.Image,
                    "prompt": str,
                },
                ...
            ]

        full=False:
            PSNR / LPIPS / MSE

        full=True:
            ImageReward / HPS / Aesthetic / PSNR / LPIPS / MSE / CLIP
        """
        if len(samples) == 0:
            raise ValueError("samples cannot be empty")

        if full and any(sample.get("prompt") is None for sample in samples):
            raise ValueError("full=True requires `prompt` for every sample.")

        # CPU: MSE / PSNR preparation -----------------------------------------
        prepared = []
        mse_scores = []
        psnr_scores = []

        for sample in samples:
            gt, pred = self._prepare_images(sample["gt_image"], sample["pred_image"])
            preserve_mask = self._prepare_preserve_mask(
                sample["mask_image"],
                sample["gt_image"].size,
            )

            mse = self._calculate_mse(gt, pred, preserve_mask)
            psnr = self._calculate_psnr(mse)

            prepared.append((gt, pred, preserve_mask))
            mse_scores.append(mse)
            psnr_scores.append(psnr)

        # LPIPS: CPU -> GPU -> 所有图片 -> CPU -------------------------------
        lpips_scores = []

        try:
            # CPU -> GPU 也放入 try。
            self._lpips_to_gpu()

            for gt, pred, preserve_mask in prepared:
                lpips_scores.append(
                    self._calculate_lpips_one(gt, pred, preserve_mask)
                )
        finally:
            self._lpips_to_cpu()

        # Light result ---------------------------------------------------------
        result = {
            "PSNR": float(np.mean(psnr_scores)),
            "LPIPS": float(np.mean(lpips_scores)),
            "MSE": float(np.mean(mse_scores)),
        }

        # full=False 到这里结束。
        if not full:
            return result

        # FULL METRICS ---------------------------------------------------------
        # 顺序:
        # CLIP GPU -> CPU
        # Aesthetic GPU -> CPU
        # ImageReward GPU -> CPU
        # HPS GPU -> CPU
        #
        # offload=True 时各评价模型不会同时常驻 GPU。
        clip_scores = self._calculate_clip_batch(samples)
        aesthetic_scores = self._calculate_aesthetic_batch(samples)
        image_reward_scores = self._calculate_image_reward_batch(samples)
        hps_scores = self._calculate_hps_batch(samples)

        # 按 BrushNet evaluation 表格顺序返回。
        return {
            "Image Reward": float(np.mean(image_reward_scores)),
            "HPS V2.1": float(np.mean(hps_scores)),
            "Aesthetic Score": float(np.mean(aesthetic_scores)),
            "PSNR": result["PSNR"],
            "LPIPS": result["LPIPS"],
            "MSE": result["MSE"],
            "CLIP Similarity": float(np.mean(clip_scores)),
        }

    # TensorBoard ==============================================================

    @staticmethod
    def log_tensorboard(writer, metrics, step, prefix="validation"):
        """
        写 TensorBoard scalar。

        示例:
            validation/PSNR
            validation/LPIPS
            validation/MSE
            validation/ImageReward
            validation/HPS_V2.1
            validation/Aesthetic
            validation/CLIP
        """
        tag_map = {
            "Image Reward": "ImageReward↑",
            "HPS V2.1": "HPS_V2.1↑",
            "Aesthetic Score": "Aesthetic↑",
            "PSNR": "PSNR↑",
            "LPIPS": "LPIPS↓",
            "MSE": "MSE↓",
            "CLIP Similarity": "CLIP↑",
        }

        for key, value in metrics.items():
            writer.add_scalar(f"{prefix}/{tag_map.get(key, key)}", value, step)

    # Debug ====================================================================

    def print_devices(self):
        """
        检查 metric model 当前设备。

        full evaluation 完成且 offload=True 时，
        所有已经加载的模型都应该显示 cpu。
        """
        models = {
            "LPIPS": self.lpips_metric,
            "CLIP": self.clip_metric,
            "Aesthetic CLIP": self.aesthetic_clip_model,
            "ImageReward": self.image_reward_model,
            "HPS": self.hps_model,
        }

        for name, model in models.items():
            if model is not None:
                print(f"{name}: {next(model.parameters()).device}")

    def print_cuda_memory(self, name=""):
        """
        查看当前 PyTorch CUDA 显存。

        allocated:
            当前真实被 Tensor/model 使用的显存。
log_tensorboard
        reserved:
            PyTorch CUDA allocator 保留的缓存显存。

        判断是否有模型没有释放时重点看 allocated。
        """
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3

        print(
            f"[CUDA] {name}: allocated={allocated:.3f} GB, "
            f"reserved={reserved:.3f} GB"
        )

    # Cleanup ==================================================================

    def unload_heavy_models(self):
        """
        彻底释放 heavy metrics，包括 CPU RAM。

        注意:
            训练期间不建议每次 full validation 后调用，
            否则下次会重新加载 CLIP/Aesthetic/ImageReward/HPS。

        推荐:
            训练期间只让 evaluator 自动 GPU -> CPU；
            整个训练结束时由 close() 调用。
        """
        if self.clip_metric is not None:
            try:
                self.clip_metric = self.clip_metric.cpu()
            except Exception:
                pass
            self.clip_metric = None

        if self.aesthetic_model is not None:
            try:
                self.aesthetic_model = self.aesthetic_model.cpu()
            except Exception:
                pass
            self.aesthetic_model = None

        if self.aesthetic_clip_model is not None:
            try:
                self.aesthetic_clip_model = self.aesthetic_clip_model.cpu()
            except Exception:
                pass
            self.aesthetic_clip_model = None

        self.aesthetic_preprocess = None

        if self.image_reward_model is not None:
            try:
                self._image_reward_to_device("cpu")
            except Exception:
                pass
            self.image_reward_model = None

        if self.hps_model is not None:
            try:
                self.hps_model = self.hps_model.cpu()
            except Exception:
                pass
            self.hps_model = None

        self.hps_preprocess = None
        self.hps_tokenizer = None

        gc.collect()
        self._clear_cuda()

    def close(self):
        """
        训练结束时调用。

        LPIPS -> CPU
        heavy metrics -> CPU -> 删除引用
        gc.collect()
        torch.cuda.empty_cache()
        """
        if self.lpips_metric is not None:
            try:
                self.lpips_metric = self.lpips_metric.cpu()
            except Exception:
                pass

        self.unload_heavy_models()

        gc.collect()
        self._clear_cuda()
