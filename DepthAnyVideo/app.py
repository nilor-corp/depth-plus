import gradio as gr
import logging
import os
import random
import shutil
import torch
from easydict import EasyDict
import numpy as np
from dav.pipelines import DAVPipeline
from dav.models import UNetSpatioTemporalRopeConditionModel
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from dav.utils import img_utils


def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Load models once to avoid reloading on every inference
def load_models(model_base, device):
    vae = AutoencoderKLTemporalDecoder.from_pretrained(model_base, subfolder="vae")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_base, subfolder="scheduler"
    )
    unet = UNetSpatioTemporalRopeConditionModel.from_pretrained(
        model_base, subfolder="unet"
    )
    unet_interp = UNetSpatioTemporalRopeConditionModel.from_pretrained(
        model_base, subfolder="unet_interp"
    )
    pipe = DAVPipeline(
        vae=vae,
        unet=unet,
        unet_interp=unet_interp,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)
    return pipe


# Load models at startup
MODEL_BASE = "hhyangcs/depth-any-video"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_TYPE)
pipe = load_models(MODEL_BASE, DEVICE)
logging.info(f"Models loaded on {DEVICE}")


def depth_any_video(
    file_path,
    denoise_steps=3,
    num_frames=32,
    decode_chunk_size=16,
    num_interp_frames=16,
    num_overlap_frames=6,
    max_resolution=1024,
):
    """
    Perform depth estimation on the uploaded video/image.
    Save the result in the output directory and return the path for display.
    """
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Replace spaces with underscores in the filename
    sanitized_file_name = os.path.basename(file_path).replace(" ", "_")
    local_input_path = os.path.join(output_dir, sanitized_file_name)
    shutil.copy(file_path, local_input_path)

    # Prepare configuration
    cfg = EasyDict(
        {
            "model_base": MODEL_BASE,
            "data_path": local_input_path,
            "output_dir": output_dir,
            "denoise_steps": denoise_steps,
            "num_frames": num_frames,
            "decode_chunk_size": decode_chunk_size,
            "num_interp_frames": num_interp_frames,
            "num_overlap_frames": num_overlap_frames,
            "max_resolution": max_resolution,
            "seed": random.randint(0, 10000),
        }
    )

    seed_all(cfg.seed)

    file_name = os.path.splitext(sanitized_file_name)[0]
    is_video = cfg.data_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    if is_video:
        num_interp_frames = cfg.num_interp_frames
        num_overlap_frames = cfg.num_overlap_frames
        num_frames = cfg.num_frames
        assert num_frames % 2 == 0, "num_frames should be even."
        assert (
            2 <= num_overlap_frames <= (num_interp_frames + 2 + 1) // 2
        ), "Invalid frame overlap."
        max_frames = (num_interp_frames + 2 - num_overlap_frames) * (
            num_frames // 2
        )
        image, fps = img_utils.read_video(cfg.data_path, max_frames=max_frames)

        if image is None or len(image) == 0:
            raise ValueError("No frames extracted from the video. Please check the input file.")
    else:
        image = img_utils.read_image(cfg.data_path)

        if image is None or len(image) == 0:
            raise ValueError("Failed to read the image. Please check the input file.")

    image = img_utils.imresize_max(image, cfg.max_resolution)
    image = img_utils.imcrop_multi(image)
    image_tensor = np.ascontiguousarray(
        [_img.transpose(2, 0, 1) / 255.0 for _img in image]
    )
    image_tensor = torch.from_numpy(image_tensor).to(DEVICE)

    with torch.no_grad(), torch.autocast(
        device_type=DEVICE_TYPE, dtype=torch.float16
    ):
        pipe_out = pipe(
            image_tensor,
            num_frames=cfg.num_frames,
            num_overlap_frames=cfg.num_overlap_frames,
            num_interp_frames=cfg.num_interp_frames,
            decode_chunk_size=cfg.decode_chunk_size,
            num_inference_steps=cfg.denoise_steps,
        )

    disparity = pipe_out.disparity
    disparity_colored = pipe_out.disparity_colored
    image = pipe_out.image
    # (N, H, 2 * W, 3)
    merged = np.concatenate(
        [
            image,
            disparity_colored,
        ],
        axis=2,
    )

    if is_video:
        output_path = os.path.join(cfg.output_dir, f"{file_name}_depth.mp4")  # Ensure .mp4 extension
        img_utils.write_video(
            output_path,
            merged,
            fps
        )
        return output_path
    else:
        output_path = os.path.join(cfg.output_dir, f"{file_name}_depth.png")
        img_utils.write_image(
            output_path,
            merged[0],
        )
        return output_path


# Define Gradio interface
title = "Depth Any Video with Scalable Synthetic Data"
description = """
Upload a video or image to perform depth estimation using the Depth Any Video model.
Adjust the parameters as needed to control the inference process.
"""

iface = gr.Interface(
    fn=depth_any_video,
    inputs=[
        gr.File(label="Upload Video/Image", type="filepath"),  # Correct type usage
        gr.Slider(1, 10, step=1, value=3, label="Denoise Steps"),
        gr.Slider(16, 64, step=1, value=32, label="Number of Frames"),
        gr.Slider(8, 32, step=1, value=16, label="Decode Chunk Size"),
        gr.Slider(8, 32, step=1, value=16, label="Number of Interpolation Frames"),
        gr.Slider(2, 10, step=1, value=6, label="Number of Overlap Frames"),
        gr.Slider(512, 2048, step=32, value=1024, label="Maximum Resolution"),
    ],
    outputs=gr.Video(label="Depth Enhanced Video/Image"),
    title=title,
    description=description,
    examples=[["demos/arch_2.jpg"], ["demos/wooly_mammoth.mp4"]],
    allow_flagging="never",
    analytics_enabled=False,
)

if __name__ == "__main__":
    iface.launch(share=True)
