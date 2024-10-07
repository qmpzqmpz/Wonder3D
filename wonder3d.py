
import torch
import requests
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
from diffusers import DiffusionPipeline, DDIMScheduler, AutoencoderTiny  # only tested on diffusers[torch]==0.19.3, may have conflicts with newer versions of diffusers
from huggingface_hub import hf_hub_download

repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SD15-1step-lora.safetensors"


def load_wonder3d_pipeline():

    pipeline = DiffusionPipeline.from_pretrained(
    'qmpzqmpz/wonder3d-v1.0', # or use local checkpoint './ckpts'
    custom_pipeline='./mvdiffusion/pipelines/pipeline.py',
    trust_remote_code=True
    )

    # enable xformers
    # pipeline.unet.enable_xformers_memory_efficient_attention()
    pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")

    #if torch.cuda.is_available():
    pipeline.to('cpu')
    return pipeline

pipeline = load_wonder3d_pipeline()

pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipeline.fuse_lora(components=['unet'])

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")


# Download an example image.
cond = Image.open("C:\\Users\\mrcam\\OneDrive\\사진\\output2.jpg")

# The object should be located in the center and resized to 80% of image height.
cond = Image.fromarray(np.array(cond)[:, :, :3])

# Run the pipeline!
images = pipeline(cond, num_inference_steps=1, output_type='pt', guidance_scale=1.0).images

result = make_grid(images, nrow=8, value_range=(0, 1))

save_image(result, 'result.png')