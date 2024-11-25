import os
import random
import shutil
import torch
from easydict import EasyDict
import numpy as np
import cv2
import gradio as gr

from DepthAnyVideo.dav.pipelines import DAVPipeline
from DepthAnyVideo.dav.models import UNetSpatioTemporalRopeConditionModel
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from DepthAnyVideo.dav.utils import img_utils

MODEL_BASE = "hhyangcs/depth-any-video"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_TYPE)

class DepthPlusDepthAnyVideo:
    def seed_all(self, seed: int = 0):
        """
        Set random seeds of all components.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load_models(self, model_base, device):
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
    
    def process_depth(
        self,
        progress=gr.Progress(),
        video_path=None,
        outdir=None,
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
        sanitized_file_name = os.path.basename(video_path).replace(" ", "_")
        local_input_path = os.path.join(output_dir, sanitized_file_name)
        shutil.copy(video_path, local_input_path)

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

        self.seed_all(cfg.seed)

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

        pipe = self.load_models(MODEL_BASE, DEVICE)
        print(f"DepthAnyVideo models loaded on {DEVICE}")

        print("Starting DepthAnyVideo processing...")
        # Get total frame count for progress reporting
        total_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {total_frames} frames...")

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
                callback=lambda step, total: progress(step/total, desc="Depth estimation")
            )
            print(f"Depth estimation complete, writing output...")

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
            # Verify the file exists and return absolute path
            if os.path.isfile(output_path):
                abs_path = os.path.abspath(output_path)
                print(f"Video created successfully at: {abs_path}")
                return [abs_path]  # Return as list to match expected format
            else:
                print(f"Error: Video file not found at expected path: {output_path}")
                return None
        else:
            output_path = os.path.join(cfg.output_dir, f"{file_name}_depth.png")
            img_utils.write_image(
                output_path,
                merged[0],
            )
            return [output_path] if os.path.isfile(output_path) else None