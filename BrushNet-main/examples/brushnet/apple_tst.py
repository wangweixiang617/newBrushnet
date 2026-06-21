from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image

# input source image / mask image path and the text prompt
# image_path="examples/brushnet/src/test_image.jpg"
# mask_path="examples/brushnet/src/test_mask.jpg"
image_path="src/test_image.jpg"
mask_path="src/test_mask.jpg"
caption="A cake on the table."

init_image = cv2.imread(image_path)[:,:,::-1]
mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
init_image = init_image * (1-mask_image)

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

generator = torch.Generator("cuda").manual_seed(1234)

init_image.save("output.png")