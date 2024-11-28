import os
import random
import shutil
import torch
from easydict import EasyDict
from utils import load_torch_file, get_bitsize_from_torch_type, make_exr, construct_output_paths
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
        mp4=True,
        png=False,
        exr=False,
        is_png_8bit=False,
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
        
        # Prepare configuration
        cfg = EasyDict(
            {
                "model_base": MODEL_BASE,
                "data_path": video_path,
                "output_dir": outdir,
                "denoise_steps": denoise_steps,
                "num_frames": num_frames,
                "decode_chunk_size": decode_chunk_size,
                "num_interp_frames": num_interp_frames,
                "num_overlap_frames": num_overlap_frames,
                "max_resolution": max_resolution,
                "seed": random.randint(0, 10000),
            }
        )

        is_video = cfg.data_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

        # You can provide a file path or a directory full of videos
        if os.path.isfile(video_path):
            if(is_video):
                print("Video file found at: ", video_path)
                filenames = [video_path]
            else:
                raise ValueError("The file is not an mp4 file")
        else:
            filenames = os.listdir(video_path)
            filenames = [os.path.join(video_path, file) for file in filenames if file.endswith(".mp4")]

        print(f"outdir: {outdir}")
        os.makedirs(outdir, exist_ok=True)

        # Replace spaces with underscores in the filename
        sanitized_file_name = os.path.basename(video_path).replace(" ", "_")
        filename = os.path.splitext(sanitized_file_name)[0]
        #local_input_path = os.path.join(output_dir, sanitized_file_name)
        #shutil.copy(video_path, local_input_path)

        self.seed_all(cfg.seed)

        # Iterate through all videos and process one by one
        mp4s_out = []
        for k, filename in enumerate(filenames):
            print(f'Processing video {k+1}/{len(filenames)}: {filename}')
            progress(0, desc="Reading video frames")
            
            # Determine the suffix based on the processing type
            model_type = "relative"     # TODO: Add support for absolute depth?
            paths = construct_output_paths(filename, outdir, "depth", basename_string_concat=model_type, is_png_8bit=is_png_8bit, is_exr_32bit=True)

            # Get video info first for progress tracking
            raw_video = cv2.VideoCapture(cfg.data_path)
            frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = raw_video.get(cv2.CAP_PROP_FPS)

            # Write each frame to PNG, EXR, and/or MP4
            if mp4:
                mp4_output_path = paths['mp4']
                #print(f"Initializing MP4 writer with path: {mp4_output_path}")
                #mp4_out = cv2.VideoWriter(mp4_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
            if png:
                png_output_path = paths['png']
            if exr:
                exr_output_path = paths['exr']
                
            total_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

            raw_video.release()  # Release it since we'll read it again with img_utils

            progress(0.2, desc="Preprocessing frames")
            
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

            image = img_utils.imresize_max(image, cfg.max_resolution)
            image = img_utils.imcrop_multi(image)
            image_tensor = np.ascontiguousarray(
                [_img.transpose(2, 0, 1) / 255.0 for _img in image]
            )
            image_tensor = torch.from_numpy(image_tensor).to(DEVICE)

            progress(0.3, desc="Loading models")
            pipe = self.load_models(MODEL_BASE, DEVICE)
            
            progress(0.4, desc="Extracting depth with DAV")
            print(f"DAV pipe processing...")
            with torch.no_grad(), torch.autocast(device_type=DEVICE_TYPE, dtype=torch.float16):
                pipe_out = pipe(
                    image_tensor,
                    num_frames=cfg.num_frames,
                    num_overlap_frames=cfg.num_overlap_frames,
                    num_interp_frames=cfg.num_interp_frames,
                    decode_chunk_size=cfg.decode_chunk_size,
                    num_inference_steps=cfg.denoise_steps,
                )
                
            disparity = pipe_out.disparity

            print(f"Processing {total_frames} frames...")

            # MP4 video output handling
            if mp4:
                progress(0.5, desc=f"Processing video")
                print(f"Writing MP4 to: {mp4_output_path}")
                depth = disparity
                depth_mp4 = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                depth_mp4 = depth_mp4.astype(np.uint8)
                depth_mp4 = np.repeat(depth_mp4[..., np.newaxis], 3, axis=-1)
                img_utils.write_video(mp4_output_path, depth_mp4, fps)
                print(f"MP4 written to {mp4_output_path}")
                
                mp4s_out.append(mp4_output_path)
                #mp4_out.release()

            if png or exr:
                for frame_count, frame in enumerate(disparity):
                    print(f"Processing frame {frame_count + 1}/{total_frames}")
                    progress((frame_count + 1) / total_frames, 
                            desc=f"Processing frame {frame_count + 1}/{total_frames}")
                    
                    depth = disparity[frame_count]
                    
                    # PNG image sequence output handling
                    if png:
                        print(f"Writing PNG frame to: {png_output_path}")
                        if is_png_8bit:
                            bitsize, nptype = get_bitsize_from_torch_type(torch.float8_e4m3fn)
                        else:
                            bitsize, nptype = get_bitsize_from_torch_type(torch.float16)
                        depth_png = (depth - depth.min()) / (depth.max() - depth.min()) * bitsize
                        depth_png = depth_png.astype(nptype)
                        png_filename = os.path.join(png_output_path, '{:04d}.png'.format(frame_count))
                        success = cv2.imwrite(png_filename, depth_png)
                        if not success:
                            print(f"Error writing {png_filename}")
                            
                    # EXR image sequence output handling
                    if exr:
                        print(f"Writing EXR frame to: {exr_output_path}")
                        depth_exr = depth.astype(np.float32) # explicity set to 32-bit float from np instead of using get_bitsize_from_torch_type
                        exr_filename = os.path.join(exr_output_path, '{:04d}.exr'.format(frame_count))

                        success = make_exr(exr_filename, depth_exr)
                        if not success:
                            print(f"Error writing {exr_filename}")
                        else:
                            print(f"Successfully wrote {exr_filename}")

        print("Depth processing complete")
        progress(1.0, desc="Complete")

        return mp4s_out