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
    

def determine_image_type(channels):
    num_channels = len(channels)
    print(f"Number of channels: {num_channels}")

    if num_channels == 1:
        print("This is a single-channel image.")
    elif num_channels == 3:
        print("This is an RGB image.")
    else:
        print("This is an image with an unexpected number of channels.")

    # Check the shape of each channel
    for i, tensor in enumerate(channels):
        print(f"Shape of channel {i}: {tensor.shape}")
        width, height = tensor.shape[-2:]

    return num_channels, width, height

    

def make_exr(filename_prefix, channels=None):
    # File path handling
    useabs = os.path.isabs(filename_prefix)
    if not useabs:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        filename_prefix = os.path.join(current_folder, filename_prefix)

    # print(f"filename_prefix: {filename_prefix}")

    # Determine channel names and prepare the data
    default_names = ["R", "G", "B", "A"] + [f"Channel{i}" for i in range(4, len(channels))]
    exr_data = {}

    for j, tensor in enumerate(channels):
        # Ensure the data is converted to float32 if necessary
        exr_data[default_names[j]] = tensor.astype(np.float32)

    writepath = filename_prefix

    # Write EXR file
    return write_exr(writepath, exr_data)

def write_exr(writepath, exr_data):
    try:
        # Determine the height and width from one of the provided channels
        height, width = list(exr_data.values())[0].shape[:2]

        # Create the EXR file header with dynamic channel names, using FLOAT for float32 data
        header = OpenEXR.Header(width, height)
        header['channels'] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for name in exr_data.keys()}

        # Create the EXR file
        exr_file = OpenEXR.OutputFile(writepath, header)

        # Convert each channel to float32 without altering the actual numerical values
        channel_data = {name: data.tobytes() for name, data in exr_data.items()}

        # Write the channel data to the EXR file
        exr_file.writePixels(channel_data)
        exr_file.close()
        
        print(f"EXR file saved successfully to {writepath}")
    except Exception as e:
        print(f"Failed to write EXR file: {e}")
