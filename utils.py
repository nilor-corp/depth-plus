import torch
import safetensors.torch
import numpy as np
import OpenEXR
import Imath
import os

def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd

def get_bitsize_from_torch_type(torch_type):
    bit8 = 255
    bit16 = 65535
    bit32 = 4294967295

    if(torch_type == torch.float8_e4m3fn or torch_type == torch.float8_e5m2 or torch_type == torch.float8_e4m3fnuz):
        return bit8, np.uint8
    elif(torch_type == torch.float16):
        return bit16, np.uint16
    elif(torch_type == torch.float32):
        return bit32, np.uint32
    else:
        raise ValueError("Invalid torch type get_bitsize_from_torch_type")

def determine_image_type(tensor):
    if len(tensor.shape) == 2:
        # Single-channel image
        num_channels = 1
        height, width = tensor.shape
    elif len(tensor.shape) == 3:
        # Multi-channel image
        num_channels = tensor.shape[2]
        height, width = tensor.shape[:2]
    else:
        raise ValueError("Unsupported tensor shape")
    return num_channels, width, height

def construct_output_paths(video_path, outdir, process_type, basename_string_concat="", is_png_8bit=True, is_exr_32bit=True):
    """
    Constructs output paths for video processing.

    :param video_path: Path to the video file.
    :param outdir: Base output directory.
    :param process_type: Type of processing ('depth', 'optical', etc.).
    :param basename_suffix: Suffix to append to the basename (e.g., '_metric', '_relative').
    :param is_png_8bit: Whether the PNG output should be 8-bit.
    :param is_exr_32bit: Whether the EXR output should be 32-bit.
    :return: A dictionary containing paths for mp4, png, and exr outputs.
    """
    
    basename = os.path.splitext(os.path.basename(video_path))[0]
    if basename_string_concat:
        process_type = f"{process_type}_{basename_string_concat}"
    video_output_dir = os.path.join(outdir, basename, process_type)
    os.makedirs(video_output_dir, exist_ok=True)

    paths = {}

    # Construct paths for each output type
    mp4_output_path = os.path.join(video_output_dir, 'mp4')
    os.makedirs(mp4_output_path, exist_ok=True)
    paths['mp4'] = os.path.join(mp4_output_path, f'{basename}_{process_type}.mp4')
    
    png_output_path = os.path.join(video_output_dir, 'png')
    png_output_path = f"{png_output_path}_8bit" if is_png_8bit else f"{png_output_path}_16bit"
    os.makedirs(png_output_path, exist_ok=True)
    paths['png'] = png_output_path
    
    exr_output_path = os.path.join(video_output_dir, 'exr')
    exr_output_path = f"{exr_output_path}_32bit" if is_exr_32bit else f"{exr_output_path}_16bit"
    os.makedirs(exr_output_path, exist_ok=True)
    paths['exr'] = exr_output_path

    return paths

def make_exr(filename_prefix, data=None):
    # File path handling
    useabs = os.path.isabs(filename_prefix)
    if not useabs:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        filename_prefix = os.path.join(current_folder, filename_prefix)

    # print(f"len(data): {len(data.shape)}")
    # print(f"data: {data}")

    # Determine if the data is single-channel or multi-channel
    num_channels, width, height = determine_image_type(data)
    print(f"width: {width}, height: {height}, num_channels: {num_channels}")

    # Prepare the data for writing
    exr_data = {}

    if num_channels == 1:
        # Single-channel image
        exr_data["R"] = data.astype(np.float32)
    elif num_channels > 1:
        # Multi-channel image
        default_names = ["R", "G", "B", "A"] + [f"Channel{i}" for i in range(4, num_channels)]
        for k in range(num_channels):
            # print(f"Channel {k}: {default_names[k]}")
            # print(f"Data shape: {data.shape}")
            # print(f"Data type: {data.dtype}")

            channel_data = data[:, :, k]
            exr_data[default_names[k]] = channel_data.astype(np.float32)
    else:
        raise ValueError("Unsupported tensor shape")

    # print(f"EXR data: {exr_data}")

    # Write the EXR file
    return write_exr(filename_prefix, exr_data, width, height)

def write_exr(writepath, exr_data, width, height):
    success = False
    try:
        # Create the EXR file header with dynamic channel names, using FLOAT for float32 data
        header = OpenEXR.Header(width, height)
        header['channels'] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for name in exr_data.keys()}

        # Create the EXR file
        exr_file = OpenEXR.OutputFile(writepath, header)

        # Convert the channel data to bytes
        channel_data = {name: data.tobytes() for name, data in exr_data.items()}

        # Write the channel data to the EXR file
        exr_file.writePixels(channel_data)
        exr_file.close()

        print(f"EXR file saved successfully to {writepath}")

        success = True
        
    except Exception as e:
        print(f"Failed to write EXR file: {e}")

    return success
