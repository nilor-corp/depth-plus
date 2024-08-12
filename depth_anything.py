import torch
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from contextlib import nullcontext
from contextlib import nullcontext
from utils import load_torch_file, get_bitsize_from_torch_type, make_exr
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
import numpy as np

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    pass

class DepthPlusDepth:
    def load_model(self, model_path, device):
        dtype = torch.float16 if "fp16" in model_path else torch.float32
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            #'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The specified path does not exist: {model_path}")
        else:
            print(f"Loading model from: {model_path}")
        if "vitl" in model_path:
            encoder = "vitl"
        elif "vitb" in model_path:
            encoder = "vitb"
        elif "vits" in model_path:
            encoder = "vits"

        if "hypersim" in model_path:
            max_depth = 20.0
        else:
            max_depth = 80.0

        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            if 'metric' in model_path:
                print("Loading depth v2 metric model")
                model = DepthAnythingV2(**{**model_configs[encoder], 'is_metric': True, 'max_depth': max_depth})
            else:
                print("Loading depth v2 relative model")
                model = DepthAnythingV2(**model_configs[encoder])
        
        state_dict = load_torch_file(model_path)
        if is_accelerate_available:
            for key in state_dict:
                set_module_tensor_to_device(model, key, device=device, dtype=dtype, value=state_dict[key])
        else:
            model = model.load_state_dict(state_dict)

        model = model.eval()
        da_model = {
            "model": model,
            "dtype": dtype,
            "is_metric": model.is_metric
        }
        return da_model

    #relative mp4 is default
    def process_depth(self, video_path=None, outdir=None, metric=False, mp4=True, png=False, exr=False, is_png_8bit=True):

        if(video_path is None):
            video_path=r"test-video"
        if(outdir is None):    
            outdir=r"test-video-output"
        if metric:
            model_path = r"models\depth_anything_v2_metric_hypersim_vitl_fp32.safetensors"
        else:
            #relative
            model_path = r"models\depth_anything_v2_vitl.pth"

        print("Processing depth")
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'
        da_model = self.load_model(model_path, device)
        print("Loaded model")
        model = da_model["model"]
        model_dtype = da_model["dtype"]

        # You can provide a file path or a directory full of videos
        if os.path.isfile(video_path):
            if(video_path.endswith(".mp4")):
                print("Video file found at: ", video_path)
                filenames = [video_path]
            else:
                raise ValueError("The file is not an mp4 file")
        else :
            filenames = os.listdir(video_path)
            filenames = [os.path.join(video_path, file) for file in filenames if file.endswith(".mp4")]

        os.makedirs(outdir, exist_ok=True)

        #iterate through all videos and process one by one
        for k, filename in enumerate(filenames):
            print(f'Progress: {k+1}/{len(filenames)}: {filename}')

            raw_video = cv2.VideoCapture(filename)
            frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
            output_width = frame_width

            filename = os.path.basename(filename)
            basename = filename[:filename.rfind('.')]
            type_dir = os.path.join("depth", basename)
            output_dir = os.path.join(outdir, type_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if(metric):
                basename = f"{basename}_metric"
            else:
                basename = f"{basename}_relative"

            if mp4:
                mp4_output_path_and_name = os.path.join(output_dir, f'{basename}_depth.mp4')
                mp4_out = cv2.VideoWriter(mp4_output_path_and_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (output_width, frame_height))
                print("Writing mp4 to: ", mp4_output_path_and_name)
            if png:
                png_output_path = os.path.join(output_dir, f'{basename}_depth_png')
                if(is_png_8bit):
                    png_output_path = f"{png_output_path}_8bit"
                else:
                    png_output_path = f"{png_output_path}_16bit"
                if not os.path.exists(png_output_path):
                    os.makedirs(png_output_path)
                print("Writing png's to: ", png_output_path)
            if exr:
                exr_output_path = os.path.join(output_dir, f'{basename}_depth_exr')
                if(model_dtype == torch.float8_e4m3fn or model_dtype == torch.float8_e5m2 or model_dtype == torch.float8_e4m3fnuz):
                    exr_output_path = f"{exr_output_path}_8bit"
                if(model_dtype == torch.float16):
                    exr_output_path = f"{exr_output_path}_16bit"
                if(model_dtype == torch.float32):
                    exr_output_path = f"{exr_output_path}_32bit"
                if not os.path.exists(exr_output_path):
                    os.makedirs(exr_output_path)
                print("Writing exr's to: ", exr_output_path)

            frame_count = 0
            while raw_video.isOpened():
                ret, frame = raw_video.read()
                if not ret:
                    break
                
                depth = model.infer_image(frame, 1024)

                if mp4:
                    #force 8 bit if mp4
                    depth_mp4 = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                    depth_mp4 = depth_mp4.astype(np.uint8) 
                    depth_mp4 = np.repeat(depth_mp4[..., np.newaxis], 3, axis=-1)
                    mp4_out.write(depth_mp4)
                if png:
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
                if exr:
                    bitsize, nptype = get_bitsize_from_torch_type(model_dtype)

                    # disable remapping for exr
                    # depth_exr = (depth - depth.min()) / (depth.max() - depth.min()) * bitsize
                    # depth_exr = depth_exr.astype(nptype)

                    # instead pass values directly
                    depth_exr = depth.astype(nptype)

                    exr_filename = os.path.join(exr_output_path, '{:04d}.exr'.format(frame_count))
                    success = make_exr(exr_filename, [depth_exr])

                
                frame_count += 1


            raw_video.release()
            if mp4:
                mp4_out.release()
        print("Depth processing complete")

            








